"""Figure 9: TPR vs true fractional cover on the stratified forced-fraction
simulated test set, (a) MLP-NPV and (b) SpecTF.

Each class is forced to fractions in 0.05-wide bins (outputs/data/
forced_fraction.npz, built from held-out endmembers); hit = probability >=
0.5, as plot_tpr_by_fraction.py (no threshold file). Shading: Wilson 95% CI.
Bins are left-open/right-closed; the last bin holds the pure pixels.

Writes figures/figure9_tpr_by_fraction.pdf (+ .png).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C


def wilson(p, n, z=1.96):
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def bin_tpr(frac, hit, w=0.05):
    edges = np.arange(0, 1 + 1e-9, w)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (frac > lo) & (frac <= hi)
        if m.any():
            rows.append(((lo + hi) / 2, hit[m].mean(), m.sum()))
    return np.array(rows).T


def main():
    C.mpl_style()
    ff = np.load(C.out_path("data", "forced_fraction.npz"))
    cls, frac = ff["forced_class"], ff["forced_fraction"]
    w = C.CFG["forced_fraction"]["bin_width"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True, constrained_layout=True)
    rows = []
    for ax, model, title in zip(axes, ["mlp", "spectf"], ["(a) MLP-NPV", "(b) SpecTF"]):
        hit = C.load_preds(model, "forced")["presence"]
        for k, name in enumerate(C.CLASSES):
            m = cls == k
            c, tpr, n = bin_tpr(frac[m], hit[m, k].astype(float), w)
            lo, hi = wilson(tpr, n)
            color = C.soften(C.CLASS_COLORS[name])
            ax.fill_between(c, lo, hi, color=color, alpha=0.2, lw=0)
            ax.plot(c, tpr, marker="o", ms=3, lw=1.2, color=color, label=name)
            rows += [dict(model=model, cls=name, center=a, tpr=b, n=int(d)) for a, b, d in zip(c, tpr, n)]
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlim(0, 1), ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_xlabel("Fractional cover")
        ax.grid(ls=":", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("True positive rate")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside right center", frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(C.fig_path(f"figure9_tpr_by_fraction.{ext}"), bbox_inches="tight")
    df = pd.DataFrame(rows)
    df.to_csv(C.tab_path("figure9_tpr_by_fraction.csv"), index=False)
    print(df.pivot_table(index="center", columns=["model", "cls"], values="tpr").round(2).to_string())


if __name__ == "__main__":
    main()
