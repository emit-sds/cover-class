"""Figure 10: active classes per pixel on the real EMIT validation set, global
vs SpecTf-conditioned E(MC)^2.

(a) Distribution of the number of classes with fraction >= 0.05.
(b) Mean (+/- 1 std) active classes vs. threshold; each pixel counts at least
    its dominant class.

Port of EMIT-FC/frac-cover/scripts/compare_sma.py plot_class_count_figure
(which replaced sma/class_count_distribution.py in db36a37): Global first,
blue/orange, all-NaN rows dropped, 16 pt fonts on a 12 x 4.6 in canvas,
bold (a)/(b) tags, shared legend on top, no titles.

Writes figures/figure10_sma_ood_class_count.png (+ .pdf) and prints the
summary numbers quoted in Section 4.3.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

STYLE = {"global": ("Global", "#4C72B0"), "spectf": ("Conditional", "#DD8452")}
PAPER_RC = {"font.size": 16, "axes.labelsize": 16, "xtick.labelsize": 15,
            "ytick.labelsize": 15, "legend.fontsize": 16}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(ls=":", alpha=0.6)
    ax.set_axisbelow(True)


def main():
    thr0 = C.CFG["presence_threshold"]
    K = len(C.CLASSES)
    ts = np.round(np.arange(0.005, 0.5001, 0.005), 4)
    summary = []
    with plt.rc_context(PAPER_RC):
        fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
        ymax = 0.0
        for v, (label, color) in STYLE.items():
            P = C.load_sma("ood", v)
            P = np.nan_to_num(P[~np.all(np.isnan(P), axis=1)])
            counts = (P >= thr0).sum(1)
            hist = np.array([(counts == k).mean() for k in range(1, K + 1)])
            edges = np.append(np.arange(1, K + 1) - 0.5, K + 0.5)
            y = np.append(hist, hist[-1])
            ymax = max(ymax, hist.max())
            a.fill_between(edges, y, step="post", color=color, alpha=0.35, lw=0)
            a.step(edges, y, where="post", color=color, lw=2.5, label=label)

            cs = np.stack([np.maximum((P >= t).sum(1), 1) for t in ts])
            mu, sd = cs.mean(1), cs.std(1)
            b.fill_between(ts, np.maximum(mu - sd, 1), mu + sd, color=color, alpha=0.2, lw=0)
            b.plot(ts, mu, color=color, lw=2.5, label=label)
            summary.append(dict(variant=label, n=len(P),
                                **{f"frac_{k}": h for k, h in zip(range(1, K + 1), hist)},
                                frac_ge3=hist[2:].sum(), frac_le2=hist[:2].sum(), frac_ge4=hist[3:].sum(),
                                **{f"mean@{t}": cs[np.argmin(abs(ts - t))].mean() for t in (0.05, 0.1, 0.2)}))

        a.set_xlim(0.5, K + 0.5), a.set_xticks(range(1, K + 1)), a.set_ylim(0, ymax * 1.15)
        a.set_xlabel(rf"Active classes per pixel (fraction $\geq$ {thr0:g})")
        a.set_ylabel("Fraction of pixels")
        b.axvline(thr0, color="gray", ls=":", lw=1)
        b.set_xlim(0, ts[-1]), b.set_ylim(bottom=0)
        b.set_xticks(np.arange(0, ts[-1] + 1e-9, 0.1))
        b.set_xlabel("Fraction threshold"), b.set_ylabel("Mean active classes per pixel")
        for ax, tag in [(a, "a"), (b, "b")]:
            _style(ax)
            ax.text(-0.14, 1.03, f"({tag})", transform=ax.transAxes, fontsize=18,
                    fontweight="bold", va="bottom", ha="left")
        h, l = a.get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, 1.08),
                   handlelength=2.5, columnspacing=2.5)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(C.fig_path(f"figure10_sma_ood_class_count.{ext}"), dpi=200, bbox_inches="tight")
    df = pd.DataFrame(summary)
    df.to_csv(C.tab_path("figure10_class_counts.csv"), index=False)
    print(df.round(3).T.to_string())


if __name__ == "__main__":
    main()
