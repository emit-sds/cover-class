"""Evaluate a saved checkpoint on the simulation-val + OOD sets.

Produces the same metric artifacts as the end-of-training report (markdown/txt
tables + ROC/confusion/regression figures) for an existing weights file, using
the same configs — no training, no W&B. This is the standalone counterpart to
the report block in ``train.py``; both go through ``engine`` so the numbers
match exactly.

    python -m models.eval --train-config models/train_config.yaml \\
        --weights models/runs/<run>/model_epoch1000.pth --outdir models/eval_out

The task, model architecture, sim/OOD data, and min_frac all come from the
train config (override the task with --task if the checkpoint was trained for a
different head). Checkpoints saved by train.py are already the schedule-free
averaged (deployable) weights, so no optimizer swap is needed here.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# pylint: disable=wrong-import-position
import rich_click as click
import numpy as np

from specmix import MixtureDataset, fractions_to_presence
from spectf.model import SpecTfEncoder
from spectf.utils import get_device

try:
    from models.ood_dataset import load_ood
    from models import reporting as R
    from models import engine as E
except ModuleNotFoundError:
    from ood_dataset import load_ood
    import reporting as R
    import engine as E


@click.command()
@click.option("--train-config", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Path to the training YAML config (data/model/loss/min_frac).")
@click.option("--weights", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Path to the model weights (.pth) to evaluate.")
@click.option("--task", default=None, type=click.Choice(["classification", "regression"]),
              help="Override the task from the config.")
@click.option("--outdir", default=None, type=click.Path(file_okay=False),
              help="Output directory (default: <weights dir>/eval).")
@click.option("--sim-val-size", default=None, type=int, help="Override sim_val_size.")
@click.option("--num-workers", default=None, type=int, help="Override num_workers.")
def main(train_config, weights, task, outdir, sim_val_size, num_workers):
    cfg = E.load_yaml(train_config)
    cfg_dir = os.path.dirname(os.path.abspath(train_config))

    task = task or cfg["task"]
    sim_config_path = E.resolve(cfg_dir, cfg["sim_config"])
    model_config_path = E.resolve(cfg_dir, cfg["model_config"])
    ood_path = E.resolve(cfg_dir, cfg["ood_path"])
    outdir = outdir or os.path.join(os.path.dirname(os.path.abspath(weights)), "eval")
    os.makedirs(outdir, exist_ok=True)

    m_cfg = E.load_yaml(model_config_path)
    sim_cfg = E.load_yaml(sim_config_path)

    batch_size = cfg["batch_size"]
    workers = (cfg["training"]["num_workers"] if num_workers is None else num_workers)
    sim_val_size = sim_val_size or cfg["sim_val_size"]
    seed = cfg.get("random_seed", 42)

    min_frac = cfg.get("min_frac")
    if min_frac is None:
        min_frac = sim_cfg["mixing"]["interclass_min_frac"]
    min_frac = float(min_frac)

    # --- data: same simulation-val split + OOD set as training -----------------
    val_ds = MixtureDataset(sim_config_path, base_seed=seed + 1, split="val",
                            split_frac=cfg.get("split_frac", 0.5),
                            split_seed=cfg.get("split_seed", 0))
    classes = val_ds.sim.classes
    n_classes = val_ds.sim.n_classes

    print(f"[data] caching {sim_val_size} simulation-val rows ...")
    sim_X, sim_frac = E.cache_sim_val(val_ds, batch_size, sim_val_size, workers)
    sim_presence = fractions_to_presence(sim_frac, min_frac)

    ood_X, ood_Y = load_ood(ood_path, classes,
                            mask_ambiguous=cfg.get("mask_ood_ambiguous", True))

    # --- model -----------------------------------------------------------------
    device = get_device()
    print(f"[device] {device}  [weights] {weights}")
    model = E.build_model(m_cfg, val_ds.wavelengths, n_classes, device)
    E.load_weights(model, weights, device)
    spec = E.make_taskspec(task, n_classes, cfg, min_frac)

    # --- inference (checkpoint is already averaged; optimizer=None) ------------
    sim_probs = E.predict_numpy(model, sim_X, device, spec, batch_size)
    ood_probs = E.predict_numpy(model, ood_X, device, spec, batch_size)

    md = E.compute_sim_ood_metrics(sim_probs, sim_presence, ood_probs, ood_Y, classes)
    sections, figs = E.sim_ood_artifacts(md, sim_probs, sim_presence, sim_frac,
                                         ood_probs, ood_Y, classes, spec.is_regression)

    basename = f"eval_{task}_{os.path.splitext(os.path.basename(weights))[0]}"
    md_path = R.write_report(sections, outdir, basename,
                             title=f"eval {task} — {os.path.basename(weights)}")
    json_path = R.write_json(sections, outdir, basename)
    fig_paths = R.save_figures(figs, outdir, basename)
    print(f"[report] wrote {md_path}, {json_path}, and {len(fig_paths)} figures to {outdir}")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
