"""Figure 6: example validation polygon (FC-35). Unchanged by v2 (no model).

Copied from EMIT-FC/paper/polygon_figure.py; only the hard-coded paths at the
top were changed to this repo. Needs the FC-35 granule (see README).
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper figure for one validation polygon:

  (a) EMIT scene RGB (orthorectified via the GLT) with the polygon boundary
      and a box marking the zoom extent; other validation polygons that fall
      inside the scene are drawn faintly.
  (b) Zoom on the polygon -- EMIT pixels by default, or an external image
      (e.g. a high-res basemap screenshot) via --polygon-image.
  (c) The polygon's spectra (individual pixels + mean ± 1 sd), with gaps at
      the dropped water bands, and the polygon's class labels
      (present / absent / unlabeled).

Inputs
  --h5       validation_*.h5  (labels, mmgis_id, spectra, attrs: classes,
             good_wavelengths). labels: 0 = absent, 1 = present, 2 = unlabeled.
  --geojson  polygons_verified_*.geojson; matched on properties["name"]
             (e.g. "FC-3") == h5 mmgis_id.
  --nc-file  EMIT_L2A_RFL_*.nc for the scene containing the polygon.

Example
  python polygon_figure.py \
      --h5 validation_20260908_subsampled.h5 \
      --geojson polygons_verified_20260908.geojson \
      --nc-file EMIT_L2A_RFL_001_20231210T..._001.nc \
      --mmgis-id FC-3 --out fig_FC-3.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon as MplPolygon, Rectangle, ConnectionPatch
from matplotlib.lines import Line2D

# -----------------------------------------------------------------------------
# Hard-coded inputs -- edit these, then just run:  python polygon_figure.py
# (command-line flags still override them if given)
# -----------------------------------------------------------------------------
_PAPER = Path(__file__).resolve().parent
H5_FILE = _PAPER.parent / "datasets" / "ood" / "validation_20260908_subsampled.h5"
GEOJSON_FILE = _PAPER.parents[1] / "data" / "polygons_verified_20260908.geojson"
NC_FILE = _PAPER / "outputs" / "downloads" / "FC-35" / "EMIT_L2A_RFL_001_20240701T235941_2418315_004.nc"
MMGIS_ID = "FC-35"
POLYGON_IMAGE = None          # e.g. Path(".../FC-3_basemap.png"), or None
SCENE_TITLE = None            # None -> uses the .nc file name
OUT_FILE = _PAPER / "figures" / f"figure6_polygon_{MMGIS_ID}.pdf"

# Same colors as plot_scene.py (kept local so this script is standalone).
CLASS_COLOR_MAP = {
    "water": "#5B9BD5",
    "pv": "#6FAE5C",
    "npv": "#D8B463",
    "soil": "#A9764F",
    "snow+ice": "#7FC3C9",
}
CLASS_DISPLAY = {"soil": "Soil", "pv": "PV", "npv": "NPV",
                 "snow+ice": "Snow/ice", "water": "Water"}
LABEL_TEXT = {0: "absent", 1: "present", 2: "uncertain"}

# EMIT band indices (0-based, full 285-band cube) for R, G, B ~ 650/560/490 nm
RGB_IDX = [39, 24, 14]
BOUNDARY_COLOR = "#FFD400"   # selected polygon / zoom box
OTHER_POLY_COLOR = "white"


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def load_validation(h5_path, mmgis_id):
    with h5py.File(h5_path, "r") as f:
        ids = f["mmgis_id"][()].astype(str)
        sel = ids == mmgis_id
        if not sel.any():
            raise KeyError(f"{mmgis_id!r} not in {h5_path}. "
                           f"Examples: {sorted(set(ids))[:10]}")
        spectra = f["spectra"][sel]
        labels = f["labels"][sel]
        classes = [c.decode() if isinstance(c, bytes) else str(c)
                   for c in np.ravel(f.attrs["classes"])]
        wl = np.ravel(f.attrs["good_wavelengths"]).astype(float)
    # labels are constant within a polygon; take the first row
    return spectra, labels[0], classes, wl


def load_polygons(geojson_path):
    """name -> list of exterior rings (lon, lat), Polygon or MultiPolygon."""
    with open(geojson_path) as f:
        gj = json.load(f)
    polys = {}
    for ft in gj["features"]:
        geom = ft["geometry"]
        if geom["type"] == "Polygon":
            rings = [np.asarray(geom["coordinates"][0])[:, :2]]
        elif geom["type"] == "MultiPolygon":
            rings = [np.asarray(p[0])[:, :2] for p in geom["coordinates"]]
        else:
            continue
        polys[ft["properties"]["name"]] = rings
    return polys


def load_ortho_rgb(nc_file, rgb_idx=RGB_IDX):
    """Read only the 3 RGB bands and orthorectify with the EMIT GLT.

    Returns rgb (rows, cols, 3) on the ortho grid and extent
    [lon_min, lon_max, lat_min, lat_max] for imshow.
    """
    ds = xr.open_dataset(nc_file)
    rfl = ds["reflectance"].isel(bands=rgb_idx).values.astype(np.float32)
    fill = ds["reflectance"].attrs.get("_FillValue", -9999)
    rfl[(rfl == fill) | (rfl < -1)] = np.nan
    gt = ds.attrs["geotransform"]
    ds.close()

    loc = xr.open_dataset(nc_file, group="location")
    glt_x = loc["glt_x"].values.astype(np.int64)
    glt_y = loc["glt_y"].values.astype(np.int64)
    loc.close()

    valid = (glt_x > 0) & (glt_y > 0)          # GLT is 1-based, 0 = no data
    rgb = np.full(glt_x.shape + (3,), np.nan, np.float32)
    rgb[valid] = rfl[glt_y[valid] - 1, glt_x[valid] - 1]

    nrow, ncol = glt_x.shape
    extent = [gt[0], gt[0] + ncol * gt[1], gt[3] + nrow * gt[5], gt[3]]
    return rgb, extent


def stretch(rgb, low=2, high=98):
    out = np.zeros_like(rgb)
    for b in range(3):
        band = rgb[..., b]
        fin = band[np.isfinite(band)]
        if fin.size == 0:
            continue
        lo, hi = np.percentile(fin, [low, high])
        out[..., b] = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)
    alpha = np.isfinite(rgb).all(-1).astype(np.float32)   # no-data -> transparent
    return np.dstack([np.nan_to_num(out), alpha])


def insert_gaps(wl, spectra, max_step=20.0):
    """Insert NaN columns where wavelengths jump (dropped bands) so lines break."""
    jumps = np.where(np.diff(wl) > max_step)[0] + 1
    wl_g = np.insert(wl, jumps, np.nan)
    sp_g = np.insert(spectra, jumps, np.nan, axis=1)
    return wl_g, sp_g


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def bbox_of(rings, pad_frac=0.35, min_pad=0.004):
    pts = np.vstack(rings)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    pad = max(pad_frac * max(x1 - x0, y1 - y0), min_pad)
    return x0 - pad, x1 + pad, y0 - pad, y1 + pad


def add_rings(ax, rings, color, lw, alpha=1.0, z=3):
    for r in rings:
        ax.add_patch(MplPolygon(r, closed=True, fill=False, edgecolor=color,
                                linewidth=lw, alpha=alpha, zorder=z))


def panel_label(ax, s):
    ax.text(0.02, 0.98, s, transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold", color="black",
            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.8),
            zorder=10)


def make_figure(h5, geojson, nc_file, mmgis_id, out, polygon_image=None,
                scene_title=None, max_lines=200, dpi=300):
    spectra, labels, classes, wl = load_validation(h5, mmgis_id)
    polys = load_polygons(geojson)
    if mmgis_id not in polys:
        raise KeyError(f"{mmgis_id!r} not found in {geojson} (properties.name)")
    rings = polys[mmgis_id]

    rgb, extent = load_ortho_rgb(nc_file)
    img = stretch(rgb)
    lon0, lon1, lat0, lat1 = extent

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    fig = plt.figure(figsize=(7.2, 6.4))           # ~2-column journal width
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.25, 1],
                  width_ratios=[1, 1], hspace=0.28, wspace=0.18)
    ax_scene = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_spec = fig.add_subplot(gs[1, :])

    # --- (a) scene -----------------------------------------------------------
    ax_scene.set_facecolor("white")
    ax_scene.imshow(img, extent=extent, interpolation="nearest")
    for name, rr in polys.items():               # other polygons in the scene
        if name == mmgis_id:
            continue
        p = np.vstack(rr)
        if ((p[:, 0] > lon0) & (p[:, 0] < lon1) &
                (p[:, 1] > lat0) & (p[:, 1] < lat1)).any():
            add_rings(ax_scene, rr, OTHER_POLY_COLOR, 0.6, alpha=0.7)
    add_rings(ax_scene, rings, BOUNDARY_COLOR, 1.2, z=4)
    zx0, zx1, zy0, zy1 = bbox_of(rings)
    ax_scene.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False,
                                 edgecolor=BOUNDARY_COLOR, lw=0.9, ls="--", zorder=5))
    ax_scene.set_xlim(lon0, lon1)
    ax_scene.set_ylim(lat0, lat1)
    ax_scene.set_xlabel("Longitude (°)")
    ax_scene.set_ylabel("Latitude (°)")
    ax_scene.set_title(scene_title or Path(nc_file).stem, fontsize=7)
    ax_scene.tick_params(labelsize=7)
    panel_label(ax_scene, "a")

    # --- (b) polygon zoom ----------------------------------------------------
    if polygon_image:
        ax_zoom.imshow(plt.imread(polygon_image))
        ax_zoom.set_xticks([]); ax_zoom.set_yticks([])
        ax_zoom.set_title("Polygon (reference imagery)", fontsize=7)
    else:
        ax_zoom.imshow(img, extent=extent, interpolation="nearest")
        add_rings(ax_zoom, rings, BOUNDARY_COLOR, 1.6, z=4)
        ax_zoom.set_xlim(zx0, zx1)
        ax_zoom.set_ylim(zy0, zy1)
        ax_zoom.set_xlabel("Longitude (°)")
        ax_zoom.tick_params(labelsize=7)
        ax_zoom.ticklabel_format(useOffset=False)
        ax_zoom.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax_zoom.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax_zoom.set_title(f"{mmgis_id} (EMIT, 60 m pixels)", fontsize=7)
        for spine in ax_zoom.spines.values():
            spine.set_edgecolor(BOUNDARY_COLOR); spine.set_linestyle("--")
            spine.set_linewidth(1.0)
        # connector lines from zoom box to panel (b)
        for (xa, ya), (xb, yb) in [((zx1, zy1), (0, 1)), ((zx1, zy0), (0, 0))]:
            fig.add_artist(ConnectionPatch(
                xyA=(xa, ya), coordsA=ax_scene.transData,
                xyB=(xb, yb), coordsB=ax_zoom.transAxes,
                color=BOUNDARY_COLOR, lw=0.7, ls="--", zorder=1))
    panel_label(ax_zoom, "b")

    # --- (c) spectra ---------------------------------------------------------
    present = [c for c, l in zip(classes, labels) if l == 1]
    color = CLASS_COLOR_MAP.get(present[0], "#444444") if present else "#444444"

    wl_g, sp_g = insert_gaps(wl, spectra)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(sp_g), min(max_lines, len(sp_g)), replace=False)
    ax_spec.plot(wl_g, sp_g[idx].T, color=color, lw=0.3, alpha=0.25)
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)   # NaN gap columns
        mu, sd = np.nanmean(sp_g, 0), np.nanstd(sp_g, 0)
    ax_spec.fill_between(wl_g, mu - sd, mu + sd, color=color, alpha=0.25, lw=0)
    ax_spec.plot(wl_g, mu, color="black", lw=1.2)
    ax_spec.set_xlim(np.nanmin(wl) - 10, np.nanmax(wl) + 10)
    ax_spec.set_ylim(bottom=0)
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel("Reflectance")
    ax_spec.grid(ls=":", lw=0.4, alpha=0.6)
    ax_spec.spines[["top", "right"]].set_visible(False)

    # label key: filled = present, open = absent, grey ? = unlabeled
    handles = []
    for c, l in zip(classes, labels):
        col = CLASS_COLOR_MAP.get(c, "#888888")
        mk = dict(marker="s", ms=7, ls="none")
        if l == 1:
            h = Line2D([], [], mfc=col, mec="black", mew=0.5, **mk)
        elif l == 0:
            h = Line2D([], [], mfc="white", mec=col, mew=1.2, **mk)
        else:
            h = Line2D([], [], mfc="#dddddd", mec="#999999", mew=0.5, **mk)
        h.set_label(f"{CLASS_DISPLAY.get(c, c)}: {LABEL_TEXT.get(int(l), l)}")
        handles.append(h)
    handles += [Line2D([], [], color=color, lw=0.8, alpha=0.6,
                       label=f"pixels (n={len(spectra)})"),
                Line2D([], [], color="black", lw=1.2, label="mean ± 1 sd")]
    ax_spec.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                   frameon=False, fontsize=7, handletextpad=0.4,
                   title=f"{mmgis_id} labels", title_fontsize=7,
                   alignment="left", borderaxespad=0)
    panel_label(ax_spec, "c")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", default=H5_FILE)
    p.add_argument("--geojson", default=GEOJSON_FILE)
    p.add_argument("--nc-file", default=NC_FILE, help="EMIT L2A RFL .nc scene")
    p.add_argument("--mmgis-id", default=MMGIS_ID, help="e.g. FC-3")
    p.add_argument("--polygon-image", default=POLYGON_IMAGE,
                   help="optional image for panel (b) instead of the EMIT zoom")
    p.add_argument("--scene-title", default=SCENE_TITLE)
    p.add_argument("--max-lines", type=int, default=200,
                   help="max individual spectra drawn (mean/sd use all)")
    p.add_argument("--out", default=OUT_FILE)
    p.add_argument("--dpi", type=int, default=300)
    a = p.parse_args()
    make_figure(a.h5, a.geojson, a.nc_file, a.mmgis_id, a.out,
                a.polygon_image, a.scene_title, a.max_lines, a.dpi)


if __name__ == "__main__":
    main()
