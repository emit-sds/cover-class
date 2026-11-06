"""Scene-map plotting shared by Figures 1, 11 and 12.

Ports of EMIT-FC scenes/plot_scene.py (plot_fraction_cover,
plot_num_classes_comparison, plot_fraction_cover_comparison): white->class
color maps, one shared sqrt-stretched (PowerNorm gamma 0.5) scale whose max is
the 99th percentile of positive fractions rounded up to 0.05.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, PowerNorm

import common as C


def load_scene(name):
    d = C.out_path("scenes", name, "x").parent
    if not (d / "fraction_map_full.npy").exists():
        raise SystemExit(f"No scene outputs in {d}; run pipeline/s05_scene.py {name} first.")
    out = dict(full=np.load(d / "fraction_map_full.npy"), rgb=np.load(d / "rgb_quicklook.npy"),
               meta=json.loads((d / "scene_meta.json").read_text()))
    if (d / "fraction_map.npy").exists():
        out["cond"] = np.load(d / "fraction_map.npy")
    return out


def shared_vmax(stack, pct=99):
    pos = stack[np.isfinite(stack) & (stack > 0)]
    if not pos.size:
        return 1.0, False
    vmax = float(np.clip(np.ceil(np.percentile(pos, pct) * 20) / 20, 0.05, 1.0))
    return vmax, bool(np.nanmax(stack) > vmax)


def _frame(ax):
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("black")


def _cmap(cls):
    return LinearSegmentedColormap.from_list(f"{cls}_cmap", ["white", C.soften(C.CLASS_COLORS[cls])])


def _scalebar(ax, h, w, km=5, px_m=60):
    npx = km * 1000 / px_m
    x0, y0 = 0.05 * w, 0.93 * h
    ax.plot([x0, x0 + npx], [y0, y0], color="white", lw=2, solid_capstyle="butt")
    ax.text(x0 + npx / 2, y0 - 0.02 * h, f"{km:g} km", color="white", ha="center", va="bottom")
    ax.set_xlim(-0.5, w - 0.5), ax.set_ylim(h - 0.5, -0.5)


def fraction_cover(fmap, rgb, path, ncols=3, panel_width=2.3, fontsize=8, gamma=0.5):
    """RGB panel + one panel per class (Figure 1)."""
    n = len(C.CLASSES) + 1
    nrows = int(np.ceil(n / ncols))
    aspect = fmap.shape[0] / fmap.shape[1]
    vmax, clipped = shared_vmax(fmap)
    ticks = np.array([0, 0.1, 0.25, 0.5, 1.0]) * vmax
    with plt.rc_context({"font.size": fontsize, "axes.linewidth": 0.6}):
        fig, axes = plt.subplots(nrows, ncols, figsize=(panel_width * ncols, (panel_width * aspect + 0.6) * nrows),
                                 constrained_layout=True)
        axes = axes.ravel()
        axes[0].imshow(np.clip(rgb, 0, 1), interpolation="nearest", rasterized=True)
        _frame(axes[0]), axes[0].set_title("RGB composite", loc="left", fontsize=fontsize + 1, pad=3)
        cb = fig.colorbar(plt.cm.ScalarMappable(), ax=axes[0], orientation="horizontal", fraction=0.05, pad=0.02)
        cb.ax.set_visible(False)
        for ax, (c, cls) in zip(axes[1:], enumerate(C.CLASSES)):
            im = ax.imshow(fmap[:, :, c], cmap=_cmap(cls), norm=PowerNorm(gamma, vmin=0, vmax=vmax),
                           interpolation="nearest", rasterized=True)
            _frame(ax), ax.set_title(f"Fractional cover: {cls}", loc="left", fontsize=fontsize + 1, pad=3)
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, ticks=ticks,
                              extend="max" if clipped else "neither")
            cb.ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2g"))
            cb.outline.set_linewidth(0.5)
            cb.ax.tick_params(width=0.5, length=2, labelsize=fontsize - 1)
        for ax in axes[n:]:
            fig.delaxes(ax)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def num_classes_comparison(rgb, maps, titles, path, threshold=0.05, width=7.0, fontsize=8):
    """RGB + class-count maps (fraction > threshold, strict, as EMIT-FC) (Figure 11)."""
    K = maps[0].shape[-1]
    levels = np.arange(1, K + 1)
    cmap = ListedColormap(plt.get_cmap("GnBu_r")(np.linspace(0.01, 0.8, len(levels))))
    cmap.set_bad("white")
    norm = BoundaryNorm(np.arange(0.5, K + 1.5), cmap.N)
    h, w = maps[0].shape[:2]
    n_pan = 1 + len(maps)
    with plt.rc_context({"font.size": fontsize, "axes.linewidth": 0.6}):
        fig, axes = plt.subplots(1, n_pan, figsize=(width, 0.9 * width / n_pan * h / w + 0.3), layout="constrained")
        axes[0].imshow(np.clip(rgb, 0, 1), interpolation="nearest", rasterized=True)
        _scalebar(axes[0], h, w)
        im = None
        for ax, m in zip(axes[1:], maps):
            valid = ~np.all(np.isnan(m), axis=2)
            counts = np.full((h, w), np.nan, np.float32)
            counts[valid] = (m[valid] > threshold).sum(-1)
            counts[counts == 0] = np.nan
            im = ax.imshow(counts, cmap=cmap, norm=norm, interpolation="nearest", rasterized=True)
        for ax, t in zip(axes, ["RGB composite"] + list(titles)):
            _frame(ax), ax.set_title(t, loc="left", fontsize=fontsize + 1, pad=3)
        cb = fig.colorbar(im, ax=list(axes), ticks=levels, fraction=0.046, pad=0.02, shrink=0.9)
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(width=0.5, length=0)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def fraction_cover_comparison(maps, titles, path, width=7.5, fontsize=7, gamma=0.5):
    """Rows = unmixing variants, columns = classes, shared scale (Figure 12)."""
    n, m = len(C.CLASSES), len(maps)
    stack = np.stack(maps)
    vmax, clipped = shared_vmax(stack)
    ticks = np.array([0, 0.25, 1.0]) * vmax
    h, w = maps[0].shape[:2]
    panel_w = width / n
    mosaic = [[f"{r}_{c}" for c in range(n)] for r in range(m)]
    with plt.rc_context({"font.size": fontsize, "axes.linewidth": 0.6}):
        fig, axd = plt.subplot_mosaic(mosaic, figsize=(width, m * 0.9 * panel_w * h / w + 0.7), layout="constrained")
        fig.get_layout_engine().set(h_pad=0.01, hspace=0)
        for c, cls in enumerate(C.CLASSES):
            col = []
            for r in range(m):
                ax = axd[f"{r}_{c}"]
                im = ax.imshow(maps[r][:, :, c], cmap=_cmap(cls), norm=PowerNorm(gamma, vmin=0, vmax=vmax),
                               interpolation="nearest", rasterized=True)
                _frame(ax)
                if r == 0:
                    ax.set_title(cls, loc="left", fontsize=fontsize + 1, pad=3)
                if c == 0:
                    ax.set_ylabel(titles[r], fontsize=fontsize + 1)
                col.append(ax)
            cb = fig.colorbar(im, ax=col, location="bottom", fraction=0.05, pad=0.02, ticks=ticks,
                              extend="max" if clipped else "neither")
            cb.ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2g"))
            cb.outline.set_linewidth(0.5)
            cb.ax.tick_params(width=0.5, length=2, labelsize=fontsize - 1)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
