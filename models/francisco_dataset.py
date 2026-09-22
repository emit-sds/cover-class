"""Francisco fractional-cover validation set (pv / npv / soil ground truth).

Real EMIT pixels with in-situ fractional-cover ground truth for THREE classes
(pv, npv, soil). Secondary validation to the OOD set — especially trustworthy
for npv, which is PV-contaminated in the OOD labels.

Two CSVs under ``datasets/francisco/`` (same convention as
``experiments/magnitude/pca_overlap.py``):
- ``emit_pixels_rfl.csv`` — per-pixel reflectance, 285-band raw grid, with a
  ``plot_num`` column (9 pixels per plot). Dropped to the 207 good bands.
- ``fraction_output.csv`` — per-plot unmixed fractions, one row per plot
  (``plot`` column). Columns include npv/pv/soil (+ shade, ignored here).

Ground truth is per-PLOT but reflectance is per-PIXEL, so we broadcast each
plot's fraction to its pixels (GT repeated per pixel). Fractions are taken RAW
(not renormalized): the model predicts 5 classes and we compare only its
pv/npv/soil outputs against Francisco's pv/npv/soil, leaving any water/snow mass
as unaccounted error.
"""

import os

import numpy as np
import pandas as pd

from specmix.bands import drop_bad_bands, DROP_WL_RANGES

# Francisco's three ground-truth classes, in a fixed order.
FRANCISCO_CLASSES = ["pv", "npv", "soil"]


def load_francisco(datasets_dir):
    """Load Francisco pixels + broadcast per-plot fractions to each pixel.

    Args:
        datasets_dir: path to the ``datasets/`` directory (contains
            ``francisco/emit_pixels_rfl.csv`` and ``fraction_output.csv``).

    Returns:
        (X, F, plots):
          X: float32 (N, 207) reflectance on the good-band grid.
          F: float32 (N, 3) RAW pv/npv/soil fractions (column order =
             FRANCISCO_CLASSES), the plot's fraction repeated for each of its
             pixels.
          plots: (N,) array of plot ids, aligned to X/F (for plot-level
             aggregation if desired).
    """
    fdir = os.path.join(datasets_dir, "francisco")
    rfl = pd.read_csv(os.path.join(fdir, "emit_pixels_rfl.csv"))
    frac = pd.read_csv(os.path.join(fdir, "fraction_output.csv"))

    wl_cols = [c for c in rfl.columns if c != "plot_num"]
    wl = np.array([float(c) for c in wl_cols])
    X_raw = rfl[wl_cols].to_numpy(dtype=np.float32)
    X = drop_bad_bands(X_raw, wl, DROP_WL_RANGES)

    # Per-plot fraction lookup (raw, not renormalized).
    frac_by_plot = {row["plot"]: np.array([row[c] for c in FRANCISCO_CLASSES],
                                          dtype=np.float32)
                    for _, row in frac.iterrows()}

    pixel_plots = rfl["plot_num"].to_numpy()
    missing = sorted(set(pixel_plots) - set(frac_by_plot))
    if missing:
        raise ValueError(f"pixels reference plots with no fraction row: {missing[:5]}"
                         f"{' ...' if len(missing) > 5 else ''}")

    F = np.stack([frac_by_plot[p] for p in pixel_plots]).astype(np.float32)
    return X, F, pixel_plots


def restrict_to_francisco(probs_5, model_classes):
    """Select the pv/npv/soil columns from a 5-class prediction array.

    Args:
        probs_5: (N, 5) model probabilities/fractions over ``model_classes``.
        model_classes: the model's class axis (e.g. the simulator's).

    Returns:
        (N, 3) array with columns ordered as FRANCISCO_CLASSES. RAW — not
        renormalized (see module docstring).
    """
    idx = [model_classes.index(c) for c in FRANCISCO_CLASSES]
    return np.asarray(probs_5)[:, idx]


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    X, F, plots = load_francisco(os.path.join(here, "datasets"))
    print(f"X {X.shape}  F {F.shape}  plots {len(np.unique(plots))} unique")
    print(f"fraction col order: {FRANCISCO_CLASSES}")
    print(f"per-class mean fraction: "
          + "  ".join(f"{c}={F[:, i].mean():.3f}"
                      for i, c in enumerate(FRANCISCO_CLASSES)))
    print(f"reflectance range: [{X.min():.4f}, {X.max():.4f}]")
