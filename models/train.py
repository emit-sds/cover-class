"""Unified training for the fractional-cover SpecTf model (classification | regression).

One entrypoint, selected by ``task`` in the train config (or ``--task``):

- **classification** — multi-label presence. Target = fractions binarized at
  ``min_frac``; sigmoid head; ``FocalLoss`` (or BCE).
- **regression** — fractional unmixing. Target = fractions; softmax head;
  ``FocalCategoricalCrossEntropy`` (or KLDiv). Presence metrics come from
  binarizing predicted/true fractions at ``min_frac``; regression-error metrics
  (MAE/RMSE/R2) are computed on the simulation-val set.

Data comes from ``specmix.MixtureDataset`` (train/val on disjoint endmember
halves) and the real OOD validation set (``models.ood_dataset.load_ood``).

Per epoch we log scalar metrics to W&B for three regimes:
  - simulation-val @ its own F1-optimal thresholds
  - OOD @ the simulation thresholds (transfer)
  - OOD @ its own F1-optimal thresholds (the OOD threshold shifts a lot)
Figures + markdown/txt tables are written to ``outdir`` at the end of training.
"""

import os

# Deterministic CUDA workspace (harmless on MPS/CPU); set before torch import.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# pylint: disable=wrong-import-position
from datetime import datetime

import rich_click as click
import numpy as np
import torch
from torch.utils.data import DataLoader
import schedulefree

from specmix import MixtureDataset, numpy_collate, fractions_to_presence
from spectf.model import SpecTfEncoder
from spectf.utils import get_device

# models.* imports work whether run as `python -m models.train` or
# `python models/train.py` (the latter adds models/ to sys.path).
try:
    from models.ood_dataset import load_ood
    from models import metrics as M
    from models import reporting as R
    from models import engine as E
except ModuleNotFoundError:  # invoked as a script from inside models/
    from ood_dataset import load_ood
    import metrics as M
    import reporting as R
    import engine as E


# --------------------------------------------------------------------- main --
@click.command()
@click.option("--train-config", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Path to the training YAML config.")
@click.option("--task", default=None, type=click.Choice(["classification", "regression"]),
              help="Override the task from the config.")
@click.option("--outdir", default=None, type=click.Path(file_okay=False),
              help="Override the output directory.")
@click.option("--epochs", default=None, type=int, help="Override total_epochs.")
@click.option("--steps-per-epoch", default=None, type=int, help="Override steps_per_epoch.")
@click.option("--sim-val-size", default=None, type=int, help="Override sim_val_size.")
@click.option("--num-workers", default=None, type=int, help="Override num_workers.")
@click.option("--wandb-mode", default=None,
              type=click.Choice(["online", "offline", "disabled"]),
              help="Override the W&B mode.")
def main(train_config, task, outdir, epochs, steps_per_epoch, sim_val_size,
         num_workers, wandb_mode):
    cfg = E.load_yaml(train_config)
    cfg_dir = os.path.dirname(os.path.abspath(train_config))

    task = task or cfg["task"]
    sim_config_path = E.resolve(cfg_dir, cfg["sim_config"])
    model_config_path = E.resolve(cfg_dir, cfg["model_config"])
    ood_path = E.resolve(cfg_dir, cfg["ood_path"])
    outdir = outdir or E.resolve(cfg_dir, cfg.get("outdir", "./runs"))
    os.makedirs(outdir, exist_ok=True)

    m_cfg = E.load_yaml(model_config_path)
    sim_cfg = E.load_yaml(sim_config_path)

    tr = cfg["training"]
    total_epochs = epochs or tr["total_epochs"]
    steps_per_epoch = steps_per_epoch or tr["steps_per_epoch"]
    workers = tr["num_workers"] if num_workers is None else num_workers
    sim_val_size = sim_val_size or cfg["sim_val_size"]
    batch_size = cfg["batch_size"]
    checkpoint_every = tr.get("checkpoint_every", 50)
    resample_each_epoch = tr.get("resample_each_epoch", True)
    seed = cfg.get("random_seed", 42)

    min_frac = cfg.get("min_frac")
    if min_frac is None:
        min_frac = sim_cfg["mixing"]["interclass_min_frac"]
    min_frac = float(min_frac)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- data ------------------------------------------------------------------
    # The loader only consumes `steps_per_epoch` batches, drawn (shuffled) from
    # [0, epoch_size). epoch_size must cover a full epoch of distinct indices.
    train_epoch_size = max(steps_per_epoch * batch_size,
                           tr.get("epoch_size", 0) or 0)
    train_ds = MixtureDataset(sim_config_path, base_seed=seed, split="train",
                              epoch_size=train_epoch_size,
                              split_frac=cfg.get("split_frac", 0.5),
                              split_seed=cfg.get("split_seed", 0))
    val_ds = MixtureDataset(sim_config_path, base_seed=seed + 1, split="val",
                            split_frac=cfg.get("split_frac", 0.5),
                            split_seed=cfg.get("split_seed", 0))
    classes = train_ds.sim.classes
    n_classes = train_ds.sim.n_classes

    # shuffle=True: each item index maps to a deterministic mixture, and we take
    # only `steps_per_epoch` batches per epoch. Without shuffling we'd retrain on
    # the same first N mixtures every epoch; shuffling draws a fresh random subset
    # of the dataset's `epoch_size` distinct mixtures each epoch.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, collate_fn=numpy_collate,
                              drop_last=True)

    print(f"[data] caching {sim_val_size} simulation-val rows ...")
    sim_val_X, sim_val_frac = E.cache_sim_val(val_ds, batch_size, sim_val_size, workers)
    sim_val_presence = fractions_to_presence(sim_val_frac, min_frac)

    ood_X, ood_Y = load_ood(ood_path, classes,
                            mask_ambiguous=cfg.get("mask_ood_ambiguous", True))
    assert ood_X.shape[1] == len(train_ds.wavelengths), (
        f"OOD has {ood_X.shape[1]} bands, sim has {len(train_ds.wavelengths)}")

    sim_val_loader = E.make_loader(sim_val_X, sim_val_frac, batch_size)
    ood_loader = E.make_loader(ood_X, ood_Y, batch_size)

    # --- model / optim / loss --------------------------------------------------
    device = get_device()
    print(f"[device] {device}")
    mp = m_cfg["model"]
    model = E.build_model(m_cfg, train_ds.wavelengths, n_classes, device)
    spec = E.make_taskspec(task, n_classes, cfg, min_frac)

    optimizer = schedulefree.AdamWScheduleFree(
        (p for p in model.parameters() if p.requires_grad),
        lr=tr["learning_rate"], warmup_steps=tr["warmup_steps"])

    # --- W&B -------------------------------------------------------------------
    import wandb
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{task}"
    run = wandb.init(
        entity=cfg["wandb"]["entity"], project=cfg["wandb"]["project"],
        name=run_name, dir=outdir,
        mode=wandb_mode or cfg["wandb"].get("mode", "online"),
        config={"task": task, "min_frac": min_frac, "batch_size": batch_size,
                "sim_val_size": sim_val_size, "loss_kind": spec.kind,
                "model": mp, "seed": seed,
                "sim_config": sim_config_path, "model_config": model_config_path},
    )

    # --- train -----------------------------------------------------------------
    md = None  # last epoch's metrics dict, reused for the final report
    for epoch in range(total_epochs):
        # Fresh mixtures every epoch: set_epoch slides the global-index window so
        # the same indices map to new spectra. "Epoch" stays a logging unit;
        # total unique spectra seen = total_epochs * steps_per_epoch * batch_size.
        # (resample_each_epoch=False keeps a fixed dataset re-shuffled each epoch.)
        if resample_each_epoch:
            train_ds.set_epoch(epoch + 1)

        model.train(); optimizer.train()
        epoch_loss, step = 0.0, 0
        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(device=device, dtype=torch.float32).unsqueeze(-1)
            batch_Y = batch_Y.to(device=device, dtype=torch.float32)
            target = spec.target(batch_Y)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = spec.loss(logits, target)
            loss.backward()
            optimizer.step()

            nats = float(loss.item())
            epoch_loss += nats
            run.log({"loss_train": nats, "step": step + epoch * steps_per_epoch})
            step += 1
            if step >= steps_per_epoch:
                break

        run.log({"loss_train_epoch": epoch_loss / max(step, 1), "epoch": epoch})

        # --- eval (schedule-free averaged weights; infer swaps + restores) ---
        sim_probs, sim_loss = E.infer(model, sim_val_loader, device, spec, optimizer)
        # OOD labels are presence, not fractions -> no meaningful regression loss.
        ood_probs, ood_loss = E.infer(model, ood_loader, device, spec, optimizer,
                                      compute_loss=not spec.is_regression)

        # Checkpoint the AVERAGED (deployable) weights, not the fast train iterate.
        if (epoch + 1) % checkpoint_every == 0 or (epoch + 1) == total_epochs:
            optimizer.eval()
            torch.save(model.state_dict(), os.path.join(outdir, f"model_epoch{epoch+1}.pth"))
            optimizer.train()

        md = E.compute_sim_ood_metrics(sim_probs, sim_val_presence,
                                       ood_probs, ood_Y, classes)

        log = {"epoch": epoch, "loss_sim_epoch": sim_loss, "loss_ood_epoch": ood_loss}
        log.update(M.flatten_metrics(md["sim"], "sim"))
        log.update(M.flatten_metrics(md["ood_simthr"], "ood_simthr"))
        log.update(M.flatten_metrics(md["ood_oodthr"], "ood_oodthr"))
        if spec.is_regression:
            reg = M.regression_error(sim_probs, sim_val_frac, classes)
            log.update(M.flatten_metrics(reg, "sim_reg"))
        run.log(log)
        print(f"[epoch {epoch+1}/{total_epochs}] train_loss={epoch_loss/max(step,1):.4f} "
              f"sim_loss={sim_loss:.4f} ood_loss={ood_loss:.4f}")

    # --- final report ----------------------------------------------------------
    print("[report] writing tables + figures ...")
    sections, figs = E.sim_ood_artifacts(md, sim_probs, sim_val_presence, sim_val_frac,
                                         ood_probs, ood_Y, classes, spec.is_regression)

    basename = f"{SpecTfEncoder.__name__}_{run_name}"
    md_path = R.write_report(sections, outdir, basename,
                             title=f"{task} — {run_name}")
    json_path = R.write_json(sections, outdir, basename)
    fig_paths = R.save_figures(figs, outdir, basename)
    print(f"[report] wrote {md_path}, {json_path}, and {len(fig_paths)} figures to {outdir}")

    # Log final artifacts to W&B (skipped cleanly when mode=disabled).
    try:
        run.log({"final/" + os.path.splitext(os.path.basename(p))[0]: wandb.Image(p)
                 for p in fig_paths})
        run.save(md_path)
    except Exception as e:  # pragma: no cover - best-effort artifact upload
        print(f"[wandb] artifact upload skipped: {e}")
    run.finish()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
