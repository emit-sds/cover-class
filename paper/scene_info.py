"""Print the scene metadata quoted in the scene-figure captions (Figures 1, 11):
granule, acquisition time, crop center and extent, RGB band wavelengths.

Port of EMIT-FC/frac-cover/frac_cover/scenes/scene_info.py. Needs
pipeline/s05_scene.py <name> to have run (reads its scene_meta.json).

    python scene_info.py FC-4
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

import common as C
from pipeline.s05_scene import RGB_IDX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", default="FC-4")
    a = ap.parse_args()
    d = C.OUT / "scenes" / a.scene
    if not (d / "scene_meta.json").exists():
        raise SystemExit(f"No {d / 'scene_meta.json'}; run pipeline/s05_scene.py {a.scene} first.")
    meta = json.loads((d / "scene_meta.json").read_text())
    nc = meta["nc_file"]
    r0, r1, c0, c1 = meta["crop_bounds"]

    ds = xr.open_dataset(nc)
    loc = xr.open_dataset(nc, group="location")
    lat = loc["lat"].values[r0:r1, c0:c1]
    lon = loc["lon"].values[r0:r1, c0:c1]
    wl = xr.open_dataset(nc, group="sensor_band_parameters")["wavelengths"].values

    print("File:", Path(nc).name)
    print("Acquired:", ds.attrs.get("time_coverage_start"))
    print(f"Crop: rows {r0}-{r1}, cols {c0}-{c1} ({r1 - r0} x {c1 - c0} px)")
    print(f"Center: {np.nanmean(lat):.3f}, {np.nanmean(lon):.3f}")
    print(f"Lat {np.nanmin(lat):.3f} to {np.nanmax(lat):.3f}, "
          f"lon {np.nanmin(lon):.3f} to {np.nanmax(lon):.3f}")
    print("RGB wavelengths (nm):", np.round(wl[RGB_IDX]))


if __name__ == "__main__":
    main()
