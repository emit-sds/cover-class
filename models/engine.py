"""Shared training/eval primitives for the fractional-cover pipeline.

Factored out of ``train.py`` so ``eval.py`` and ``eval_francisco.py`` reuse the
exact same task dispatch, model construction, and inference — no drift between
how a model is trained and how it is later scored.

- ``TaskSpec`` — target construction, loss, and head activation per task.
- ``build_model`` / ``load_weights`` — construct a SpecTfEncoder and load a
  checkpoint saved by ``train.py`` (already the schedule-free *averaged*,
  deployable weights).
- ``infer`` / ``predict_numpy`` — run the model over a DataLoader or a raw numpy
  array, returning per-class probabilities/fractions.
- ``cache_sim_val`` — materialize a fixed simulation-val set.
- ``compute_sim_ood_metrics`` / ``sim_ood_artifacts`` — the standard sim-val +
  OOD metric dicts, tables, and figures used by both train and eval.
"""

import os

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from specmix import numpy_collate, fractions_to_presence
from spectf.model import SpecTfEncoder

try:
    from models.losses import FocalLoss, FocalCategoricalCrossEntropy
    from models.ood_dataset import OODDataset
    from models import metrics as M
except ModuleNotFoundError:  # invoked from inside models/
    from losses import FocalLoss, FocalCategoricalCrossEntropy
    from ood_dataset import OODDataset
    import metrics as M


# ------------------------------------------------------------ config helpers --
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(base_dir, path):
    """Resolve `path` relative to base_dir unless it is already absolute."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base_dir, path))


# ---------------------------------------------------------------- task setup --
class TaskSpec:
    """Task-specific bits: target construction, head activation, loss."""

    def __init__(self, task, n_classes, min_frac, focal_alpha, focal_gamma):
        self.task = task
        self.n_classes = n_classes
        self.min_frac = min_frac
        alpha = None if str(focal_alpha) == "None" else float(focal_alpha)
        self.is_regression = (task == "regression")

        if task == "classification":
            if alpha is None and focal_gamma == 0.0:
                self.criterion = nn.BCEWithLogitsLoss()
                self.kind = "bce"
            else:
                self.criterion = FocalLoss(alpha=alpha, gamma=focal_gamma)
                self.kind = "focal"
        elif task == "regression":
            if alpha is None and focal_gamma == 0.0:
                self.criterion = nn.KLDivLoss(reduction="batchmean")
                self.kind = "kldiv"
            else:
                self.criterion = FocalCategoricalCrossEntropy(alpha=alpha, gamma=focal_gamma)
                self.kind = "focal"
        else:
            raise ValueError(f"unknown task {task!r}; use classification|regression")

    def target(self, fractions):
        """Training target from a fractions tensor."""
        if self.is_regression:
            return fractions
        return (fractions > self.min_frac).to(fractions.dtype)

    def loss(self, logits, target):
        if self.kind == "kldiv":
            return self.criterion(F.log_softmax(logits, dim=-1), target)
        return self.criterion(logits, target)

    def probs(self, logits):
        """Map logits -> per-class probabilities/fractions for metrics."""
        if self.is_regression:
            return torch.softmax(logits, dim=-1)
        return torch.sigmoid(logits)


def make_taskspec(task, n_classes, cfg, min_frac):
    """Build a TaskSpec from a training-config dict's loss block."""
    return TaskSpec(task, n_classes, min_frac,
                    cfg["loss"]["focal_alpha"], cfg["loss"]["focal_gamma"])


# ------------------------------------------------------------ model plumbing --
def build_model(model_cfg, wavelengths, n_classes, device):
    """Construct a SpecTfEncoder with the banddef from `wavelengths`."""
    banddef = torch.tensor(np.asarray(wavelengths), dtype=torch.float32, device=device)
    mp = model_cfg["model"]
    model = SpecTfEncoder(banddef, dim_output=n_classes, num_heads=mp["num_heads"],
                          dim_proj=mp["dim_proj"], dim_ff=mp["dim_ff"],
                          dropout=mp["dropout"], agg=mp["agg"],
                          use_residual=mp["use_residual"],
                          num_layers=mp["num_layers"]).to(device)
    return model


def load_weights(model, weights_path, device):
    """Load a checkpoint (deployable averaged weights) into `model`."""
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    return model


# ---------------------------------------------------------------- inference --
@torch.no_grad()
def infer(model, loader, device, spec, optimizer=None, compute_loss=True):
    """Run the model over a loader, return (probs[N,C], mean_loss or nan).

    Loss is only meaningful when the targets match the task's loss:
    classification masks NaN (ambiguous OOD) entries elementwise; regression
    loss is only computed against real fractions (the simulation-val set), so
    the OOD regression loss is left as nan by passing ``compute_loss=False``.

    NOTE: schedule-free optimizers hold a fast train iterate and a separate
    averaged iterate; the averaged weights are the ones to evaluate. Callers
    training with such an optimizer must pass it so we can swap into eval
    (averaged) weights here and restore train weights afterwards. When loading a
    saved checkpoint (already averaged), pass ``optimizer=None``.
    """
    model.eval()
    if optimizer is not None:
        optimizer.eval()
    out = []
    loss_sum, nb = 0.0, 0
    for batch_X, batch_Y in loader:
        batch_X = batch_X.to(device=device, dtype=torch.float32).unsqueeze(-1)
        logits = model(batch_X)
        if compute_loss:
            target = spec.target(batch_Y.to(device=device, dtype=torch.float32))
            if spec.is_regression:
                loss_sum += float(spec.loss(logits, target).item())
                nb += 1
            else:
                mask = ~torch.isnan(target)
                if mask.any():
                    loss_sum += float(spec.loss(logits[mask], target[mask]).item())
                    nb += 1
        out.append(spec.probs(logits).cpu().numpy())
    if optimizer is not None:
        optimizer.train()
    mean_loss = loss_sum / nb if nb > 0 else float("nan")
    return np.concatenate(out, axis=0), mean_loss


@torch.no_grad()
def predict_numpy(model, X, device, spec, batch_size=512):
    """Per-class probabilities/fractions for a raw (N, bands) numpy array."""
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(np.asarray(X[i:i + batch_size]), dtype=torch.float32,
                          device=device).unsqueeze(-1)
        out.append(spec.probs(model(xb)).cpu().numpy())
    return np.concatenate(out, axis=0)


def cache_sim_val(sim_val_ds, batch_size, size, workers):
    """Materialize a fixed simulation-val set (spectra + true fractions)."""
    loader = DataLoader(sim_val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, collate_fn=numpy_collate)
    Xs, Fs, n = [], [], 0
    for X, Frac in loader:
        Xs.append(X.numpy()); Fs.append(Frac.numpy())
        n += len(X)
        if n >= size:
            break
    X = np.concatenate(Xs)[:size]
    Frac = np.concatenate(Fs)[:size]
    return X.astype(np.float32), Frac.astype(np.float32)


def make_loader(X, Y, batch_size):
    """DataLoader over cached (X, Y) numpy arrays via OODDataset + numpy_collate."""
    return DataLoader(OODDataset(X, Y), batch_size=batch_size, shuffle=False,
                      num_workers=0, collate_fn=numpy_collate)


# ------------------------------------------------ sim + OOD metric artifacts --
def compute_sim_ood_metrics(sim_probs, sim_presence, ood_probs, ood_Y, classes):
    """The three standard binary-metric regimes + the thresholds used.

    Returns a dict with per-class metric dicts under 'sim', 'ood_simthr',
    'ood_oodthr' and the threshold lists 'sim_thr', 'ood_thr'. The OOD threshold
    is recomputed on OOD (it shifts a lot from the simulation threshold).
    """
    sim_thr, _ = M.f1_opt_thr(sim_probs, sim_presence)
    ood_thr, _ = M.f1_opt_thr(ood_probs, ood_Y)
    return {
        "sim_thr": sim_thr, "ood_thr": ood_thr,
        "sim": M.binary_metrics_at(sim_probs, sim_presence, sim_thr, classes),
        "ood_simthr": M.binary_metrics_at(ood_probs, ood_Y, sim_thr, classes),
        "ood_oodthr": M.binary_metrics_at(ood_probs, ood_Y, ood_thr, classes),
    }


def sim_ood_artifacts(md, sim_probs, sim_presence, sim_frac,
                      ood_probs, ood_Y, classes, is_regression):
    """Build (sections, figs) for the sim-val + OOD report from a metrics dict.

    `md` is the output of ``compute_sim_ood_metrics``.
    """
    sections = [
        ("sim", "Simulation-val @ simulation F1-optimal threshold", md["sim"], "binary"),
        ("ood_simthr", "OOD @ simulation threshold (transfer)", md["ood_simthr"], "binary"),
        ("ood_oodthr", "OOD @ OOD F1-optimal threshold", md["ood_oodthr"], "binary"),
    ]
    figs = {
        "sim_roc": M.roc_curve_fig(sim_probs, sim_presence, classes,
                                   "ROC — simulation-val"),
        "sim_confusion": M.confusion_fig(
            (sim_probs >= np.array(md["sim_thr"])).astype(int), sim_presence, classes,
            "Confusion — simulation-val @ sim threshold"),
        "ood_roc": M.roc_curve_fig(ood_probs, ood_Y, classes, "ROC — OOD"),
        "ood_confusion": M.confusion_fig(
            (ood_probs >= np.array(md["ood_thr"])).astype(int), ood_Y, classes,
            "Confusion — OOD @ OOD threshold"),
    }
    if is_regression:
        reg = M.regression_error(sim_probs, sim_frac, classes)
        sections.append(("sim_reg", "Regression error — simulation-val", reg, "regression"))
        figs["regression_summary"] = M.regression_summary_fig(sim_probs, sim_frac, classes)
    return sections, figs
