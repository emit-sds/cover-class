"""Figure 7: t-SNE of simulated (v2), real EMIT OOD and reduced-library spectra,
colored by class.

Follows EMIT-FC tsne_comparison.plot_class_colored_embedding as called for the
paper figure (EMIT-FC db36a37): <=3000 points per source (rng seed 0), joint
StandardScaler over raw reflectance, sklearn t-SNE (perplexity 40, euclidean,
PCA init, random_state 0), per-class KDE fill (bw_adjust 1.5, no contour
lines), no title, use_paper_style() fonts (serif, 11 pt), 300 dpi.
Simulated/OOD spectra are colored by their dominant class; for OOD that is the
argmax of the 0/1/2 label row, as in the original (see --ood-class).

Writes figures/figure7_tsne_class_colored.png.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import common as C

MARKERS = {"Simulated": "X", "OOD": "o", "Library": "^"}


def ood_classes(Y, mode):
    """'argmax' reproduces the original (uncertain=2 outranks present=1);
    'present' takes the argmax over present labels only."""
    return Y.argmax(1) if mode == "argmax" else (Y == 1).argmax(1)


def sources(ood_mode, classes=None):
    X, F, _ = C.load_sim_test()
    Xo, Yo, _, _ = C.load_ood()
    Xl, ll, _ = C.load_library()
    src = {"Simulated": (X, np.array(C.CLASSES)[F.argmax(1)]),
           "OOD": (Xo, np.array(C.CLASSES)[ood_classes(Yo, ood_mode)]),
           "Library": (Xl, ll)}
    if classes is not None:
        src = {k: (x[np.isin(c, classes)], c[np.isin(c, classes)]) for k, (x, c) in src.items()}
    return src


def embed(src, max_points=3000, perplexity=40, seed=0):
    rng = np.random.default_rng(seed)
    xs, cs, ss = [], [], []
    for name, (x, c) in src.items():
        if len(x) > max_points:
            idx = rng.choice(len(x), max_points, replace=False)
            x, c = x[idx], c[idx]
        xs.append(x), cs.append(c), ss.append(np.full(len(x), name))
    x = StandardScaler().fit_transform(np.vstack(xs))
    perplexity = min(perplexity, max(5, (len(x) - 1) // 3))
    z = TSNE(n_components=2, perplexity=perplexity, init="pca", metric="euclidean",
             random_state=seed).fit_transform(x)
    return z, np.concatenate(cs), np.concatenate(ss)


def paper_style(base=11):
    """EMIT-FC tsne_comparison.use_paper_style."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": base, "axes.titlesize": base + 1,
        "axes.labelsize": base, "xtick.labelsize": base - 1, "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1, "legend.title_fontsize": base, "figure.titlesize": base + 2,
    })


def plot(z, cls, src, ax, legend="outside", density=True, bw_adjust=1.5):
    import seaborn as sns
    order = sorted(set(cls))
    if density:
        for c in order:
            m = cls == c
            if m.sum() >= 5:
                sns.kdeplot(x=z[m, 0], y=z[m, 1], fill=True, alpha=0.18, levels=8, thresh=0.15,
                            bw_adjust=bw_adjust, color=C.CLASS_COLORS[c], ax=ax, zorder=0,
                            warn_singular=False)
    for s in [s for s in MARKERS if s in set(src)]:
        lib = s == "Library"
        for c in order:
            m = (src == s) & (cls == c)
            ax.scatter(z[m, 0], z[m, 1], s=22 if lib else 8, alpha=0.75 if lib else 0.45,
                       marker=MARKERS[s], color=C.CLASS_COLORS[c], zorder=3 if lib else 1,
                       edgecolors="black" if lib else "none", linewidths=0.3)
    h_cls = [Line2D([], [], marker="o", ls="", color=C.CLASS_COLORS[c], markersize=7, label=c) for c in order]
    h_src = [Line2D([], [], marker=MARKERS[s], ls="", color="gray", markersize=7,
                    label=f"{s} (n={(src == s).sum()})") for s in MARKERS if s in set(src)]
    if legend == "outside":
        leg = ax.legend(handles=h_cls, title="Class", loc="upper left", bbox_to_anchor=(1.02, 1.0),
                        frameon=False)
        ax.add_artist(leg)
        ax.legend(handles=h_src, title="Source", loc="upper left",
                  bbox_to_anchor=(1.02, 1.0 - 0.06 * (len(order) + 2)), frameon=False)
    else:  # "inside": both legends in the axes, on a translucent box
        box = dict(frameon=True, framealpha=0.85, edgecolor="none", borderaxespad=0.6)
        leg = ax.legend(handles=h_cls, title="Class", loc="upper right", **box)
        ax.add_artist(leg)
        ax.legend(handles=h_src, title="Source", loc="lower right", **box)
    ax.set_xlabel("dim 1"), ax.set_ylabel("dim 2")
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood-class", choices=["argmax", "present"], default="argmax")
    args = ap.parse_args()
    paper_style()
    z, cls, src = embed(sources(args.ood_class))
    np.savez_compressed(C.out_path("figures", "tsne_class.npz"), z=z, cls=cls, src=src)
    fig, ax = plt.subplots(figsize=(9.5, 6.5), constrained_layout=True)
    plot(z, cls, src, ax)
    out = C.fig_path("figure7_tsne_class_colored.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
