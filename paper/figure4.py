"""Figure 4: example v2 simulated spectra from the held-out sim test set,
spanning single-class, binary and complex (3+ class) mixtures. One column per
mixture type, two example rows; each panel has a stacked bar of the true class
fractions above the spectrum.

Port of EMIT-FC/frac-cover/scripts/plot_simulated_examples.py (same example
picker and layout). The manuscript used selection seed 6 (sim_spectra_seed6),
which is the default here; the picks differ because the sim set is new. Use
--browse 0-9 to render quick looks for several seeds and choose one.

Writes figures/figure4_simulated_spectra.pdf (+ .png).
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import common as C
from figure3 import with_gaps

TITLES = {"soil": "Soil", "pv": "PV", "npv": "NPV", "snow+ice": "Snow / ice", "water": "Water"}
TYPE_TITLES = {1: "Single class", 2: "Binary mixture", 3: "Complex mixture (3+ classes)"}
CLASS_ORDER = ["soil", "pv", "npv", "snow+ice", "water"]


def pick_examples(F, n_per_type=2, min_frac=1e-6, seed=0):
    """Row indices per mixture type: distinct class combinations first, and for
    mixtures prefer ones whose smallest component is >= 10% (as EMIT-FC)."""
    rng = np.random.default_rng(seed)
    present = F > min_frac
    n_present = present.sum(1)
    picks = {}
    for k in (1, 2, 3):
        idx = np.flatnonzero(n_present >= 3) if k == 3 else np.flatnonzero(n_present == k)
        if k > 1 and idx.size:
            smallest = np.where(present[idx], F[idx], np.inf).min(1)
            good = idx[smallest >= 0.10]
            idx = good if good.size >= n_per_type else idx
        idx = rng.permutation(idx)
        chosen, seen = [], set()
        for i in idx:
            combo = tuple(np.flatnonzero(present[i]))
            if combo not in seen:
                chosen.append(int(i)), seen.add(combo)
            if len(chosen) == n_per_type:
                break
        for i in idx:
            if len(chosen) == n_per_type:
                break
            if int(i) not in chosen:
                chosen.append(int(i))
        picks[k] = chosen
    return picks


def plot(X, F, wl, seed=0, n_per_type=2):
    picks = pick_examples(F, n_per_type=n_per_type, seed=seed)
    order = [C.CLASSES.index(c) for c in CLASS_ORDER]
    fig, axes = plt.subplots(n_per_type, 3, figsize=(7.2, 4.8), constrained_layout=True, squeeze=False)
    fig.get_layout_engine().set(h_pad=0.12)
    letters = iter("abcdefghij")
    for col, k in enumerate((1, 2, 3)):
        for row in range(n_per_type):
            ax = axes[row, col]
            bar = ax.inset_axes([0, 1.04, 1, 0.16])
            bar.set_axis_off()
            if row >= len(picks[k]):
                ax.text(0.5, 0.5, "no example", ha="center", va="center", transform=ax.transAxes, color="0.5")
                ax.set_axis_off()
                continue
            i = picks[k][row]
            left = 0.0
            for j in order:
                v = float(F[i, j])
                if v <= 1e-6:
                    continue
                bar.barh(0, v, left=left, height=1.0, color=C.CLASS_COLORS[C.CLASSES[j]],
                         edgecolor="white", linewidth=0.6)
                if v >= 0.12:
                    bar.text(left + v / 2, 0, f"{v:.2f}", ha="center", va="center", fontsize=6.5)
                left += v
            bar.set_xlim(0, max(1.0, left)), bar.set_ylim(-0.5, 0.5)

            w, y = with_gaps(wl, X[i])
            ax.plot(w, y, color="black", lw=1.0)
            ax.set_xlim(wl.min(), wl.max())
            ax.set_ylim(min(0, np.nanmin(y)), np.nanmax(y) * 1.08)
            ax.grid(alpha=0.25, ls=":", lw=0.5)
            if row == 0:
                bar.set_title(f"({next(letters)}) {TYPE_TITLES[k]}", loc="left", pad=3, fontsize=8.5)
            if col == 0:
                ax.set_ylabel("Reflectance")
            if row == n_per_type - 1:
                ax.set_xlabel("Wavelength (nm)")
            else:
                ax.tick_params(labelbottom=False)
    fig.legend(handles=[Patch(facecolor=C.CLASS_COLORS[c], label=TITLES[c]) for c in CLASS_ORDER],
               loc="outside right center", frameon=False, title="True class\nfraction",
               alignment="left", labelspacing=0.8)
    print(f"seed {seed}: rows", {TYPE_TITLES[k]: v for k, v in picks.items()})
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=6, help="example-selection seed (manuscript: 6)")
    ap.add_argument("--browse", metavar="SEEDS",
                    help="quick-look PNGs for several seeds, e.g. '0-9' or '0,4,7'")
    a = ap.parse_args()
    C.mpl_style()
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    X, F, wl = C.load_sim_test()
    if a.browse:
        seeds = []
        for part in a.browse.split(","):
            lo, _, hi = part.partition("-")
            seeds += list(range(int(lo), int(hi or lo) + 1))
        for s in seeds:
            fig = plot(X, F, wl, seed=s)
            fig.savefig(C.out_path("figures", "figure4_browse", f"seed{s}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
        return
    fig = plot(X, F, wl, seed=a.seed)
    for ext in ("pdf", "png"):
        fig.savefig(C.fig_path(f"figure4_simulated_spectra.{ext}"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
