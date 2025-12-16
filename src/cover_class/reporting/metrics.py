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
from sklearn.metrics import roc_curve, auc
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
        cmo = cm(y[:, c], y_hat[:, c])
        pct = cmo / cmo.sum() * 100
        ax.imshow(cmo, cmap="Blues")
        ax.set_title(class_names[c])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for (i, j), v in np.ndenumerate(cmo):
            this_color = "black" if pct[i,j]<33 else "white"
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

    aucs = {c: auc(*roc_curve(y[:,c], y_hat[:,c])[:2]) for c in range(y.shape[1])}
    for c, A in sorted(aucs.items(), key=lambda x: x[1], reverse=True):
        fpr, tpr, _ = roc_curve(y[:,c], y_hat[:,c])
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

    ax.set(title="Miss-Class Confusion", xlabel="Predicted", ylabel="True (missed)", xticks=range(n_classes), yticks=range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                v = masked_miss_confusion_mat[i, j]
                ax.text(j, i, f"{float(v):.1f}%", ha="center", va="center", color="white" if im.norm(v) > .5 else "black", fontsize=9)

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
        yt = y[:, i]
        yp = y_hat[:, i]

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
        tp = ((y[:,c]==1)&(y_hat[:,c]==1)).sum()
        fp = ((y[:,c]==0)&(y_hat[:,c]==1)).sum()
        fn = ((y[:,c]==1)&(y_hat[:,c]==0)).sum()

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
