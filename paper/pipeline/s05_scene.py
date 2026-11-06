"""Stage 5: run a crop of an EMIT L2A RFL scene through SpecTf + E(MC)^2.

Mirrors EMIT-FC scenes/run_scene.py: crop in sensor geometry, drop bad bands,
classify valid pixels with SpecTf at the OOD (F1-opt) thresholds, then unmix
every valid pixel globally (all classes) and conditionally (predicted classes;
pixels with no predicted class are left unprocessed -> NaN, as run_scene did).

Writes outputs/scenes/<name>/{fraction_map.npy, fraction_map_full.npy,
rgb_quicklook.npy, class_probabilities.npy, scene_meta.json}.

Scenes are configured in configs/scenes.yaml; download the granule first
(see README). Usage:  python pipeline/s05_scene.py FC-8
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C

RGB_IDX = [39, 24, 14]   # ~650/560/490 nm on the 285-band grid
BAD_BANDS = [[0, 440], [1310, 1490], [1770, 2050], [2440, 2880]]


def scene_config(name):
    with open(C.PAPER_DIR / "configs" / "scenes.yaml") as f:
        cfg = yaml.safe_load(f)[name]
    pattern = os.path.join(C.PAPER_DIR / "configs", os.path.expanduser(cfg["nc_file"]))
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit(f"{name}: no granule matches {pattern}; download it first (see README).")
    cfg["nc_file"] = Path(hits[0]).resolve()
    return cfg


def crop_bounds(nrow, ncol, size, origin):
    h, w = (size, size) if np.isscalar(size) else size
    h, w = min(h, nrow), min(w, ncol)
    r0, c0 = origin if origin is not None else ((nrow - h) // 2, (ncol - w) // 2)
    r0, c0 = int(np.clip(r0, 0, nrow - h)), int(np.clip(c0, 0, ncol - w))
    return r0, r0 + h, c0, c0 + w


def clahe_stretch(rgb, clip_limit=5.0, tile=(8, 8)):
    """Per-band min-max + CLAHE (EMIT-FC plot_scene._clahe_stretch)."""
    import cv2
    rgb = rgb.astype(np.float32).copy()
    mins, maxs = np.nanmin(rgb, axis=(0, 1)), np.nanmax(rgb, axis=(0, 1))
    span = np.where(maxs - mins == 0, 1, maxs - mins)
    rgb = np.nan_to_num(np.clip((rgb - mins) / span * 255, 0, 255)).astype("uint8")
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile)
    for b in range(3):
        rgb[..., b] = clahe.apply(rgb[..., b])
    return rgb.astype(np.float32) / 255.0


def main(name, n_jobs=None):
    cfg = scene_config(name)
    out = C.out_path("scenes", name, "x").parent
    ds = xr.open_dataset(cfg["nc_file"])
    wl = xr.open_dataset(cfg["nc_file"], group="sensor_band_parameters")["wavelengths"].values
    cube = ds["reflectance"].values
    r0, r1, c0, c1 = crop_bounds(*cube.shape[:2], cfg["crop_size"], cfg.get("crop_origin"))
    cube = cube[r0:r1, c0:c1]
    h, w, _ = cube.shape
    good = np.ones(len(wl), bool)
    for lo, hi in BAD_BANDS:
        good &= ~((wl >= lo) & (wl <= hi))
    flat = cube.reshape(-1, cube.shape[-1])[:, good].astype(np.float32)
    valid = ~np.any(flat <= -9990, axis=1) & ~np.all(flat == 0, axis=1)
    print(f"{name}: crop [{r0}:{r1}, {c0}:{c1}] -> {h}x{w}, {valid.sum()} valid px, {good.sum()} bands")

    np.save(out / "rgb_quicklook.npy", clahe_stretch(cube[:, :, RGB_IDX]))
    (out / "scene_meta.json").write_text(json.dumps(
        dict(nc_file=str(cfg["nc_file"]), crop_bounds=[r0, r1, c0, c1]), indent=2))

    model, device = C.load_spectf(wl[good])
    probs = np.zeros((len(flat), len(C.CLASSES)), np.float32)
    probs[valid] = C.predict_spectf(model, device, flat[valid])
    thr = C.load_preds("spectf", "ood")["thresholds"]
    presence = (probs >= thr[None]) & valid[:, None]
    np.save(out / "class_probabilities.npy", probs.reshape(h, w, -1))

    truth = np.zeros((len(flat), len(C.CLASSES)))
    full = np.full((len(flat), len(C.CLASSES)), np.nan)
    idx = np.flatnonzero(valid)
    full[idx] = C.run_sma(flat[idx], wl[good], truth[idx], out / "sma_global", n_jobs=n_jobs)
    np.save(out / "fraction_map_full.npy", full.reshape(h, w, -1))

    cond = np.full_like(full, np.nan)
    idx = np.flatnonzero(presence.any(1))
    cond[idx] = C.run_sma(flat[idx], wl[good], truth[idx], out / "sma_conditional",
                          subsets=C.subsets_from_presence(presence[idx]), n_jobs=n_jobs)
    np.save(out / "fraction_map.npy", cond.reshape(h, w, -1))
    print(f"{name}: {len(idx)} px with >=1 detected class; done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="scene key in configs/scenes.yaml, e.g. FC-8")
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args()
    main(args.name, args.n_jobs)
