from typing import Union, Tuple, Dict, List
import numpy as np
from numpy.typing import NDArray
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
from matplotlib import cm as color_map
from matplotlib.colors import Normalize
import matplotlib.ticker as mtick
from sklearn.metrics import confusion_matrix as cm #type: ignore[import]
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import math

from cover_class.reporting.utils import make_numpy

def confusion_matrix(
        y_hat: Union[Tensor, NDArray],
        y: Union[Tensor, NDArray],
        class_names: List[str],
    ) -> Figure:
    """!
    @note: `y_hat` and `y` are (N, C) one-hot multi-label
    """
    assert len(class_names) == y.shape[1]
    y_hat = make_numpy(y_hat)
    y     = make_numpy(y)

    n_classes = y.shape[1]
    cols = 3
    rows = math.ceil(n_classes/cols)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    fig.suptitle("Confusion Matrix")
    axes = axes.flatten()

    for c in range(n_classes):
        ax = axes[c]
        mask = ~np.isnan(y[:, c])
        cmo = cm(y[mask, c], y_hat[mask, c], labels=[0, 1])
        row_totals = cmo.sum(axis=1, keepdims=True)
        pct = np.divide(
            cmo * 100,
            row_totals,
            out=np.zeros_like(cmo, dtype=float),
            where=row_totals != 0,
        )
        
        norm = Normalize(vmin=cmo.min(), vmax=cmo.max())
        cmap = plt.get_cmap("Blues")
        im = ax.imshow(cmo, cmap=cmap, norm=norm)
        
        ax.set_title(class_names[c])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"])
        
        for (i, j), v in np.ndenumerate(cmo):
            # Calculate luminance of the background color
            rgba = cmap(norm(v))
            luminance = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
            this_color = "black" if luminance > 0.5 else "white"
            ax.text(j, i, f"{v}\n({pct[i,j]:.1f}%)", ha="center", va="center", color=this_color)

    fig.tight_layout()
    return fig

def roc_auc(
        y_hat: Union[Tensor, NDArray],
        y: Union[Tensor, NDArray],
        class_names: List[str],
    ) -> Tuple[Figure, Dict]:
    """!
    @note: `y_hat` and `y` are (N, C) one-hot multi-label

    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    @todo: look at OvR vs OvO
    """
    assert len(class_names) == y.shape[1]
    y_hat = make_numpy(y_hat)
    y     = make_numpy(y)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)

    aucs = {}
    for c in range(y.shape[1]):
        mask = ~np.isnan(y[:, c])
        aucs[c] = auc(*roc_curve(y[mask, c], y_hat[mask, c])[:2])

    for c, A in sorted(aucs.items(), key=lambda x: x[1], reverse=True):
        mask = ~np.isnan(y[:, c])
        fpr, tpr, _ = roc_curve(y[mask, c], y_hat[mask, c])
        ax.plot(fpr, tpr, label=f"{class_names[c]} (AUC={A:.3f})")

    ax.plot([0,1],[0,1],"k--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, {class_names[k]: {"ROC-AUC": round(v, 3)} for k,v in aucs.items()}

def missed_class_confusion(
        y_hat: Union[Tensor, NDArray],
        y: Union[Tensor, NDArray],
        class_names: List[str],
    ) -> Figure:
    """!
    @note: `y_hat` and `y` are (N, C) one-hot multi-label

    Reference: https://github.jpl.nasa.gov/marchett/EMIT-FC/blob/main/research/plots_classifier.py#L138#L202
    """
    assert len(class_names) == y.shape[1]
    y_hat = make_numpy(y_hat)
    y     = make_numpy(y)

    n_classes = y.shape[1]
    conf = np.full((n_classes, n_classes), np.nan)

    for i in range(n_classes):
        miss = (y[:, i] == 1) & (y_hat[:, i] == 0)
        if not miss.any(): 
            continue
        miss_predictions = y_hat[miss].copy() #type: ignore
        miss_predictions[:, i] = 0
        summed_miss_predictions = miss_predictions.sum(1)
        mask = summed_miss_predictions > 0
        if mask.any():
            conf[i] = (miss_predictions[mask].T / summed_miss_predictions[mask]).T.mean(0) * 100
        else:
            conf[i] = 0

    np.fill_diagonal(conf, np.nan)
    masked_miss_confusion_mat = np.ma.masked_invalid(conf)

    fig, ax = plt.subplots(figsize=(6.5 + .3*n_classes, 6.5 + .3*n_classes))
    cmap = color_map.Reds.copy() #type: ignore
    cmap.set_bad("#D9D9D9")
    vmax = max(1.0, np.nanmax(conf))
    im = ax.imshow(masked_miss_confusion_mat, cmap=cmap, norm=Normalize(0, vmax))

    ax.set(title="Miss-Class Confusion", xlabel="Predicted Class", ylabel="True Class (Missed)", xticks=range(n_classes), yticks=range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                v = masked_miss_confusion_mat[i, j]
                if np.ma.is_masked(v) or np.isnan(v):
                    continue
                
                # Calculate luminance of the background color
                rgba = cmap(im.norm(v))
                luminance = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                text_color = "black" if luminance > 0.5 else "white"
                
                ax.text(j, i, f"{float(v):.1f}%", ha="center", va="center", color=text_color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% misses")
    cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(vmax))

    fig.tight_layout()
    return fig

def tpr_fpr(
        y_hat: Union[Tensor, NDArray],
        y: Union[Tensor, NDArray],
        class_names: List[str],
    ) -> Dict:
    """!
    @note: `y_hat` and `y` are (N, C) one-hot multi-label
    """
    assert len(class_names) == y.shape[1]
    y_hat = make_numpy(y_hat)
    y     = make_numpy(y)

    out: dict = {name:{} for name in class_names}
    for i, name in enumerate(class_names):
        mask = ~np.isnan(y[:, i])
        yt = y[mask, i]
        yp = y_hat[mask, i]

        tp = int(((yt == 1) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        out[name] = {"TPR": round(tpr, 3), "FPR": round(fpr, 3)}
    return out

def f_beta_scores(
        y_hat: Union[Tensor, NDArray],
        y: Union[Tensor, NDArray],
        class_names: List[str],
        beta:float = 1.,
    ) -> Dict:
    """!
    @note: `y_hat` and `y` are (N, C) one-hot multi-label
    """
    assert len(class_names) == y.shape[1]
    y_hat = make_numpy(y_hat)
    y     = make_numpy(y)

    out: dict = {name:{} for name in class_names}
    for c, name in enumerate(class_names):
        mask = ~np.isnan(y[:, c])
        yt = y[mask, c]
        yp = y_hat[mask, c]

        tp = ((yt == 1) & (yp == 1)).sum()
        fp = ((yt == 0) & (yp == 1)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()

        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f_b  = (1 + (beta**2)) * prec * rec / ((beta**2) * prec + rec + 1e-12)
        out[name] = {f"F-{beta} Score": round(float(f_b),3)}
    return out

def nll(): ...
def brier_score(): ...
def pr_auc_curve(): ...
def expected_calibration_error(): ...
def adaptive_calibration_error(): ...

def fpr_opt_thr(
        y_hat: Union[Tensor, NDArray],
        y: Union[Tensor, NDArray],
        target_fpr: float = 0.05,
    ) -> List[float]:
    """
    Finds the per-class threshold that results in a False Positive Rate closest to target_fpr.
    Returns a list of thresholds, one per class.
    """
    y_hat = make_numpy(y_hat)
    y = make_numpy(y)
    n_classes = y.shape[1]
    opt_thr = []
    for c in range(n_classes):
        mask = ~np.isnan(y[:, c])
        yt = y[mask, c]
        yp = y_hat[mask, c]

        # Edge case: class absent from ground truth or predictions are constant.
        if np.nansum(yt) == 0 or np.unique(yp).size == 1:
            opt_thr.append(1.0)
            continue

        fpr, tpr, thresholds = roc_curve(yt, yp)
        
        # Find the threshold where fpr is closest to target_fpr
        idx = np.argmin(np.abs(fpr - target_fpr))
        opt_thr.append(float(thresholds[idx]))
        
    return opt_thr

def f1_opt_thr(
        y_hat: Union[Tensor, NDArray],
        y: Union[Tensor, NDArray],
        beta: float = 1.0,
    ) -> List[float]:
    """
    Finds the per-class threshold that maximises the F-beta score.
    Returns a list of optimal thresholds, one per class.
    """
    y_hat = make_numpy(y_hat)
    y = make_numpy(y)
    n_classes = y.shape[1]
    opt_thr = []
    for c in range(n_classes):
        mask = ~np.isnan(y[:, c])
        yt = y[mask, c]
        yp = y_hat[mask, c]

        # Edge case: class absent from ground truth or predictions are constant.
        if np.nansum(yt) == 0 or np.unique(yp).size == 1:
            opt_thr.append(1.0)
            continue

        precision, recall, thresholds = precision_recall_curve(yt, yp)
        
        # The precision and recall arrays have one more element than the thresholds array.
        # The last precision is 1 and recall is 0.
        precision = precision[:-1]
        recall = recall[:-1]
        
        fb = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
        
        idx = np.argmax(fb)
        opt_thr.append(float(thresholds[idx]))
        
    return opt_thr
