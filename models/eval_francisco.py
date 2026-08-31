"""Evaluate a saved checkpoint on the Francisco fractional-cover set.

Francisco provides in-situ pv/npv/soil fractions for real EMIT pixels (see
``models/francisco_dataset.py``). The model predicts 5 classes; we compare only
its pv/npv/soil outputs (RAW, not renormalized) against Francisco's, per pixel
(each plot's ground-truth fraction broadcast to its 9 pixels).

    python -m models.eval_francisco --train-config models/train_config.yaml \\
        --weights models/runs/<run>/model_epoch1000.pth

Produces (written to the eval dir):
- **regression** — the primary use. Per-class MAE/RMSE/R2 table + the
  predicted-vs-true fraction summary figure, on pv/npv/soil.
- **classification** — presence metrics on pv/npv/soil. NOTE: every Francisco
  plot contains all three classes, so there are no true negatives; FPR/ROC-AUC
  are undegenerate/undefined and only TPR (recall) is meaningful. This is
  reported for completeness with a caveat in the table title.

Task selection: --task (default: the config's task). ``both`` runs whichever
apply given the checkpoint's head — pass the actual head the weights were
trained with; a classification checkpoint only has a sigmoid head, a regression
checkpoint only softmax.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# pylint: disable=wrong-import-position
import rich_click as click
import numpy as np

from spectf.utils import get_device

try:
    from models import metrics as M
    from models import reporting as R
    from models import engine as E
    from models.francisco_dataset import (load_francisco, restrict_to_francisco,
                                          FRANCISCO_CLASSES)
except ModuleNotFoundError:
    import metrics as M
    import reporting as R
    import engine as E
    from francisco_dataset import (load_francisco, restrict_to_francisco,
                                   FRANCISCO_CLASSES)


def _regression_section(probs3, F, figs):
    """Regression error table + summary figure on pv/npv/soil."""
    reg = M.regression_error(probs3, F, FRANCISCO_CLASSES)
    figs["francisco_regression"] = M.regression_summary_fig(
        probs3, F, FRANCISCO_CLASSES,
        title="Fractional regression — Francisco (per-pixel, raw pv/npv/soil)")
    return ("francisco_reg", "Regression error — Francisco (per-pixel, raw pv/npv/soil)",
            reg, "regression")


def _classification_section(probs3, F, min_frac, figs, thresholds=None):
    """Presence metrics on pv/npv/soil at the given (or F1-optimal) thresholds.

    All Francisco plots contain every class => no true negatives; FPR is 0 and
    ROC-AUC is NaN (dropped by flatten). TPR/recall is the meaningful column.

    Because every plot is all-positive, self-optimizing the threshold is
    degenerate (it collapses to ~0 to force TPR=1). Pass ``thresholds`` (per
    class, in FRANCISCO_CLASSES order) to score Francisco under a real deployed
    decision rule — e.g. the sim or OOD F1-optimal thresholds from a train/eval
    report. Falls back to self-optimized thresholds when None.
    """
    presence = (F > min_frac).astype(np.float32)
    if thresholds is None:
        thr, _ = M.f1_opt_thr(probs3, presence)
    else:
        thr = list(thresholds)
    met = M.binary_metrics_at(probs3, presence, thr, FRANCISCO_CLASSES)
    figs["francisco_confusion"] = M.confusion_fig(
        (probs3 >= np.array(thr)).astype(int), presence, FRANCISCO_CLASSES,
        "Confusion — Francisco (all-present; TPR only meaningful)")
    return ("francisco_cls",
            "Presence metrics — Francisco (all classes present; TPR only meaningful)",
            met, "binary")


@click.command()
@click.option("--train-config", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Path to the training YAML config (data/model/loss/min_frac).")
@click.option("--weights", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Path to the model weights (.pth) to evaluate.")
@click.option("--task", default=None,
              type=click.Choice(["classification", "regression", "both"]),
              help="Which metrics to produce (default: the config's task).")
@click.option("--outdir", default=None, type=click.Path(file_okay=False),
              help="Output directory (default: <weights dir>/eval).")
@click.option("--thresholds", default=None,
              help="Comma-separated presence thresholds in FRANCISCO_CLASSES "
                   f"order ({','.join(FRANCISCO_CLASSES)}) for classification. "
                   "Pass the sim or OOD F1-optimal thresholds from a "
                   "train/eval report to score Francisco under a real deployed "
                   "decision rule. Omit to self-optimize (degenerate here — all "
                   "Francisco pixels are positive, so it collapses to ~0).")
def main(train_config, weights, task, outdir, thresholds):
    cfg = E.load_yaml(train_config)
    cfg_dir = os.path.dirname(os.path.abspath(train_config))

    task = task or cfg["task"]
    thr_list = None
    if thresholds is not None:
        thr_list = [float(t) for t in thresholds.split(",")]
        if len(thr_list) != len(FRANCISCO_CLASSES):
            raise click.BadParameter(
                f"--thresholds needs {len(FRANCISCO_CLASSES)} values in "
                f"{FRANCISCO_CLASSES} order, got {len(thr_list)}")
    sim_config_path = E.resolve(cfg_dir, cfg["sim_config"])
    model_config_path = E.resolve(cfg_dir, cfg["model_config"])
    # datasets/ dir: the parent of the OOD file's directory (datasets/ood/*.h5).
    datasets_dir = os.path.dirname(os.path.dirname(E.resolve(cfg_dir, cfg["ood_path"])))
    outdir = outdir or os.path.join(os.path.dirname(os.path.abspath(weights)), "eval")
    os.makedirs(outdir, exist_ok=True)

    m_cfg = E.load_yaml(model_config_path)
    sim_cfg = E.load_yaml(sim_config_path)
    batch_size = cfg["batch_size"]

    min_frac = cfg.get("min_frac")
    if min_frac is None:
        min_frac = sim_cfg["mixing"]["interclass_min_frac"]
    min_frac = float(min_frac)

    # The model class axis comes from the simulator config (its canonical order).
    model_classes = list(sim_cfg["classes"])
    n_classes = len(model_classes)
    missing = [c for c in FRANCISCO_CLASSES if c not in model_classes]
    if missing:
        raise ValueError(f"model classes {model_classes} lack Francisco classes {missing}")

    # --- data ------------------------------------------------------------------
    X, F, plots = load_francisco(datasets_dir)
    print(f"[data] Francisco: {X.shape[0]} pixels, {len(np.unique(plots))} plots")

    # --- model -----------------------------------------------------------------
    device = get_device()
    print(f"[device] {device}  [weights] {weights}")
    # Banddef from the simulator's good-band wavelengths (matches the 207 grid).
    from specmix import MixtureSimulator
    wavelengths = MixtureSimulator(sim_config_path).wavelengths
    assert X.shape[1] == len(wavelengths), (
        f"Francisco has {X.shape[1]} bands, model expects {len(wavelengths)}")

    model = E.build_model(m_cfg, wavelengths, n_classes, device)
    E.load_weights(model, weights, device)

    # --- inference: run each head the requested task(s) need -------------------
    want_reg = task in ("regression", "both")
    want_cls = task in ("classification", "both")
    sections, figs = [], {}

    if want_reg:
        spec_reg = E.make_taskspec("regression", n_classes, cfg, min_frac)
        probs5 = E.predict_numpy(model, X, device, spec_reg, batch_size)
        probs3 = restrict_to_francisco(probs5, model_classes)
        sections.append(_regression_section(probs3, F, figs))
    if want_cls:
        spec_cls = E.make_taskspec("classification", n_classes, cfg, min_frac)
        probs5 = E.predict_numpy(model, X, device, spec_cls, batch_size)
        probs3 = restrict_to_francisco(probs5, model_classes)
        sections.append(_classification_section(probs3, F, min_frac, figs,
                                                 thresholds=thr_list))

    basename = f"francisco_{task}_{os.path.splitext(os.path.basename(weights))[0]}"
    md_path = R.write_report(sections, outdir, basename,
                             title=f"Francisco eval ({task}) — {os.path.basename(weights)}")
    json_path = R.write_json(sections, outdir, basename)
    fig_paths = R.save_figures(figs, outdir, basename)
    print(f"[report] wrote {md_path}, {json_path}, and {len(fig_paths)} figures to {outdir}")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
