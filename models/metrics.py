"""Standalone evaluation metrics for fractional-cover models.

Ported from the old ``cover_class.reporting.metrics`` but with NO dependency on
that package — pure numpy + scikit-learn + matplotlib. Everything operates on
``(N, C)`` arrays of per-class scores / labels and masks NaN entries per class
(the OOD set uses NaN for ambiguous labels; see ``models/ood_dataset.py``).

Two families:

- **Binary presence metrics.** ``f1_opt_thr`` finds the per-class F1-optimal
  probability threshold; ``binary_metrics_at`` scores predictions at a given set
  of thresholds (F1, TPR, FPR, ROC-AUC, and the 2x2 confusion counts).
- **Figures.** ``roc_curve_fig`` (all classes overlaid) and ``confusion_fig``
  (per-class 2x2 grid) for classification; ``regression_error`` /
  ``regression_summary_fig`` for the fractional-regression task.

Scores (`y_hat`) are class probabilities in [0, 1] (sigmoid for classification,
softmax for regression). Labels (`y`) are 0/1 with optional NaN.
"""

from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


# --------------------------------------------------------------- thresholds --
def f1_opt_thr(y_hat, y, beta: float = 1.0):
    """Per-class threshold maximizing the F-beta score, plus that score.

    Returns (thresholds, f_scores): two lists of length C. NaN labels are
    masked per class. For a degenerate class (no positives, or constant
    scores) the threshold is 1.0 and the score 0.0.
    """
    y_hat = np.asarray(y_hat, dtype=float)
    y = np.asarray(y, dtype=float)
    n_classes = y.shape[1]
    thresholds, scores = [], []
    for c in range(n_classes):
        mask = ~np.isnan(y[:, c])
        yt, yp = y[mask, c], y_hat[mask, c]
        if np.nansum(yt) == 0 or np.unique(yp).size == 1:
            thresholds.append(1.0)
            scores.append(0.0)
            continue
        precision, recall, thr = precision_recall_curve(yt, yp)
        # precision/recall have one extra element vs thresholds (the (1, 0) tail)
        precision, recall = precision[:-1], recall[:-1]
        fb = ((1 + beta**2) * precision * recall
              / (beta**2 * precision + recall + 1e-12))
        idx = int(np.argmax(fb))
        thresholds.append(float(thr[idx]))
        scores.append(float(fb[idx]))
    return thresholds, scores


# ----------------------------------------------------------- scalar metrics --
def binary_metrics_at(y_hat, y, thresholds, class_names: List[str]) -> Dict[str, dict]:
    """Per-class binary metrics at the given thresholds.

    Returns ``{class_name: {threshold, f1, tpr, fpr, roc_auc, tp, fp, fn, tn}}``.
    ROC-AUC is threshold-independent (computed from the raw scores); the rest
    are computed at the supplied per-class threshold. NaN labels masked.
    """
    y_hat = np.asarray(y_hat, dtype=float)
    y = np.asarray(y, dtype=float)
    assert len(thresholds) == y_hat.shape[1] == len(class_names)

    out: Dict[str, dict] = {}
    for c, name in enumerate(class_names):
        mask = ~np.isnan(y[:, c])
        yt, yp = y[mask, c], y_hat[mask, c]
        yb = (yp >= thresholds[c]).astype(int)

        tp = int(((yt == 1) & (yb == 1)).sum())
        fn = int(((yt == 1) & (yb == 0)).sum())
        fp = int(((yt == 0) & (yb == 1)).sum())
        tn = int(((yt == 0) & (yb == 0)).sum())

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)

        # ROC-AUC needs both classes present in the labels
        if np.unique(yt).size < 2:
            roc = float("nan")
        else:
            fpr_c, tpr_c, _ = roc_curve(yt, yp)
            roc = float(auc(fpr_c, tpr_c))

        out[name] = {
            "threshold": float(thresholds[c]),
            "f1": round(float(f1), 4),
            "tpr": round(float(tpr), 4),
            "fpr": round(float(fpr), 4),
            "roc_auc": round(roc, 4) if np.isfinite(roc) else roc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
    return out


def regression_error(y_hat_frac, y_frac, class_names: List[str]) -> Dict[str, dict]:
    """Per-class fractional error over the full set: MAE, RMSE, R^2.

    ``y_hat_frac`` and ``y_frac`` are (N, C) fraction arrays (rows ~sum to 1).
    R^2 is the coefficient of determination against the per-class mean.
    """
    y_hat_frac = np.asarray(y_hat_frac, dtype=float)
    y_frac = np.asarray(y_frac, dtype=float)
    out: Dict[str, dict] = {}
    for c, name in enumerate(class_names):
        yt, yp = y_frac[:, c], y_hat_frac[:, c]
        err = yp - yt
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        out[name] = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4) if np.isfinite(r2) else r2,
        }
    return out


# ------------------------------------------------------------------ figures --
_PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#e34948", "#4a3aa7",
            "#eb6834", "#008300", "#e87ba4"]
_INK = "#0b0b0b"
_GRID = "#e1e0d9"


def roc_curve_fig(y_hat, y, class_names: List[str], title: str = "ROC"):
    """ROC curves for all classes overlaid, AUC in the legend (sorted desc)."""
    y_hat = np.asarray(y_hat, dtype=float)
    y = np.asarray(y, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 6))

    aucs = {}
    for c in range(len(class_names)):
        mask = ~np.isnan(y[:, c])
        yt, yp = y[mask, c], y_hat[mask, c]
        aucs[c] = auc(*roc_curve(yt, yp)[:2]) if np.unique(yt).size >= 2 else float("nan")

    for rank, (c, a) in enumerate(sorted(aucs.items(),
                                         key=lambda kv: (np.isnan(kv[1]), -kv[1]))):
        mask = ~np.isnan(y[:, c])
        yt, yp = y[mask, c], y_hat[mask, c]
        if np.unique(yt).size < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, yp)
        ax.plot(fpr, tpr, color=_PALETTE[rank % len(_PALETTE)], lw=2,
                label=f"{class_names[c]} (AUC={a:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(title, color=_INK, loc="left")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, color=_GRID, lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def confusion_fig(y_hat_binary, y, class_names: List[str], title: str = "Confusion"):
    """Grid of per-class binary confusion matrices (counts + row %)."""
    import math
    from matplotlib.colors import Normalize

    y_hat_binary = np.asarray(y_hat_binary, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(class_names)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle(title)
    axes = np.atleast_1d(axes).ravel()

    cmap = plt.get_cmap("Blues")
    for c in range(n):
        ax = axes[c]
        mask = ~np.isnan(y[:, c])
        yt = y[mask, c].astype(int)
        yb = y_hat_binary[mask, c].astype(int)
        mat = np.zeros((2, 2), dtype=int)
        for t in (0, 1):
            for p in (0, 1):
                mat[t, p] = int(((yt == t) & (yb == p)).sum())
        row_tot = mat.sum(axis=1, keepdims=True)
        pct = np.divide(mat * 100.0, row_tot, out=np.zeros_like(mat, dtype=float),
                        where=row_tot != 0)
        norm = Normalize(vmin=mat.min(), vmax=max(1, mat.max()))
        ax.imshow(mat, cmap=cmap, norm=norm)
        ax.set_title(class_names[c])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        for (i, j), v in np.ndenumerate(mat):
            rgba = cmap(norm(v))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, f"{v}\n({pct[i, j]:.1f}%)", ha="center", va="center",
                    color="black" if lum > 0.5 else "white")
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.tight_layout()
    return fig


def regression_summary_fig(y_hat_frac, y_frac, class_names: List[str],
                           title: str = "Fractional regression (simulation-val)"):
    """Per-class predicted-vs-true fraction scatter + MAE/RMSE/R^2 annotation.

    The single 'all relevant metrics' figure for the regression task.
    """
    import math

    y_hat_frac = np.asarray(y_hat_frac, dtype=float)
    y_frac = np.asarray(y_frac, dtype=float)
    err = regression_error(y_hat_frac, y_frac, class_names)

    n = len(class_names)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle(title)
    axes = np.atleast_1d(axes).ravel()

    for c, name in enumerate(class_names):
        ax = axes[c]
        yt, yp = y_frac[:, c], y_hat_frac[:, c]
        ax.hexbin(yt, yp, gridsize=40, cmap="viridis", mincnt=1,
                  extent=(0, 1, 0, 1), bins="log")
        ax.plot([0, 1], [0, 1], "r--", lw=1)
        m = err[name]
        r2 = m["r2"]
        r2s = f"{r2:.3f}" if isinstance(r2, float) and np.isfinite(r2) else "n/a"
        ax.set_title(f"{name}\nMAE={m['mae']:.3f} RMSE={m['rmse']:.3f} R²={r2s}",
                     fontsize=9)
        ax.set_xlabel("True fraction"); ax.set_ylabel("Predicted fraction")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------- flattening --
def flatten_metrics(metrics: Dict[str, dict], prefix: str) -> Dict[str, float]:
    """Flatten ``{class: {metric: val}}`` to ``{prefix/class/metric: val}``.

    Non-finite / non-numeric values are dropped so W&B logs clean scalar
    series. Confusion counts (tp/fp/fn/tn) are kept — they are numeric.
    """
    flat: Dict[str, float] = {}
    for cls, d in metrics.items():
        for k, v in d.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                flat[f"{prefix}/{cls}/{k}"] = float(v)
    return flat
