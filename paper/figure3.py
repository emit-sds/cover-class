"""Figure 3: reduced reference spectral library per class after v2 subsampling
(train split, the E(MC)^2 library). One panel per class; retained endmembers
as thin lines, class mean in bold. Bad bands are shown as gaps.

Port of EMIT-FC/frac-cover/scripts/plot_reduced_library.py. That script reads
the per-class k-medoids HDF5s; here the library comes from the v2 simulator
(outputs/data/unmixing_library.csv), and the soil rows that came from the rock
file (Tetracorder) are drawn in the darker rock shade, as in the original.

Writes figures/figure3_reduced_library.pdf (+ .png).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D

import common as C

ROCK_COLOR = "#5E3F27"   # darker shade of the soil brown
TITLES = {"soil": "Soil", "pv": "PV", "npv": "NPV", "snow+ice": "Snow / ice", "water": "Water"}
PANEL_ORDER = ["soil", "pv", "npv", "snow+ice", "water"]


def with_gaps(wl, y, factor=3.0):
    """Insert NaNs where the band spacing jumps (dropped bands) so lines break."""
    gaps = np.where(np.diff(wl) > factor * np.median(np.diff(wl)))[0] + 1
    return np.insert(wl, gaps, np.nan), np.insert(y, gaps, np.nan, axis=-1)


def rock_mask(X, labels):
    """True for soil library rows that are spectra from the rock file."""
    from specmix.simulator import MixtureSimulator
    with open(C.CFG["sim_config"]) as f:
        cfg = yaml.safe_load(f)
    rock = [p for p in cfg["endmembers"]["soil"] if "rock" in os.path.basename(p)]
    if not rock:
        return np.zeros(len(X), bool)
    R, _, _ = MixtureSimulator._load_class(rock, cfg["drop_wl_ranges"], os.path.dirname(C.CFG["sim_config"]))
    d = np.abs(X[:, None, :] - R[None].astype(np.float32)).max(-1).min(-1)
    return (labels == "soil") & (d < 1e-6)


def main():
    C.mpl_style()
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    X, labels, wl = C.load_library()
    is_rock = rock_mask(X, labels)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, constrained_layout=True)
    axes = axes.ravel()
    for i, cls in enumerate(PANEL_ORDER):
        ax = axes[i]
        m = labels == cls
        parts = [("soil", m & ~is_rock), ("rock", m & is_rock)] if cls == "soil" else [(None, m)]
        parts = [(src, pm) for src, pm in parts if pm.any()]
        for src, pm in parts:
            w, s = with_gaps(wl, X[pm])
            ax.plot(w, s.T, color=ROCK_COLOR if src == "rock" else C.CLASS_COLORS[cls],
                    lw=0.5, alpha=0.35)
        w, mu = with_gaps(wl, X[m].mean(0))
        ax.plot(w, mu, color="black", lw=1.5)
        n_txt = " + ".join(f"{pm.sum()} {src}" if src else f"{pm.sum()}" for src, pm in parts)
        ax.set_title(f"({chr(97 + i)}) {TITLES[cls]}  (n = {n_txt})", loc="left")
        ax.set_ylim(min(0.0, X[m].min()), X[m].max() * 1.05)
        ax.set_xlim(wl.min(), wl.max())
        ax.grid(alpha=0.25, ls=":", lw=0.5)
        if i % 3 == 0:
            ax.set_ylabel("Reflectance")
        if i >= 2:
            ax.set_xlabel("Wavelength (nm)")
            ax.xaxis.set_tick_params(labelbottom=True)
        print(f"{cls}: {n_txt}")

    axes[5].axis("off")
    axes[5].legend(handles=[
        Line2D([], [], color="0.45", lw=0.8, label="Retained endmembers\n(v2 subsample)"),
        Line2D([], [], color=ROCK_COLOR, lw=0.8, label="Rock spectra in soil"),
        Line2D([], [], color="black", lw=1.5, label="Class mean"),
    ], loc="center", frameon=False, labelspacing=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(C.fig_path(f"figure3_reduced_library.{ext}"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
