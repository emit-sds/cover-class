"""Run a trained checkpoint over an EMIT L2A reflectance scene.

Reads the scene's `reflectance` (rows, cols, 285 raw bands), drops the same
bad bands the simulator drops (`specmix.bands.drop_bad_bands`) so the band
axis matches the model's banddef, runs inference pixel-by-pixel in batches,
and writes a per-class fraction/probability GeoTIFF (band per class, no
georeferencing — raw swath pixel space, matching the reference
`cover_class/models/spectf/scene_spectf.py`).

    python -m models.pred_scene --train-config models/train_config.yaml \\
        --weights models/runs/<run>/model_epoch1000.pth \\
        --scene /path/to/EMIT_L2A_RFL_..._.nc --outdir models/pred_out
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# pylint: disable=wrong-import-position
import h5py
import numpy as np
import rich_click as click
from osgeo import gdal
from tqdm import tqdm

from specmix.bands import drop_bad_bands
from spectf.utils import get_device

try:
    from models import engine as E
except ModuleNotFoundError:  # invoked as a script from inside models/
    import engine as E


def load_scene(nc_path, drop_wl_ranges):
    """Read `reflectance` + band wavelengths from an EMIT L2A .nc, drop bad bands.

    Returns (rfl_flat[N, n_bands], (n_rows, n_cols)).
    """
    with h5py.File(nc_path, "r") as f:
        rfl = np.array(f["reflectance"][:])
        wl = np.array(f["sensor_band_parameters"]["wavelengths"][:])
    n_rows, n_cols, _ = rfl.shape
    rfl = rfl.reshape(n_rows * n_cols, rfl.shape[2])
    rfl = drop_bad_bands(rfl, wl, drop_wl_ranges)
    return rfl.astype(np.float32), (n_rows, n_cols)


@click.command()
@click.option("--train-config", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Path to the training YAML config (data/model/loss/min_frac).")
@click.option("--weights", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Path to the model weights (.pth) to run.")
@click.option("--scene", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Path to the EMIT L2A reflectance scene (.nc).")
@click.option("--task", default=None, type=click.Choice(["classification", "regression"]),
              help="Override the task from the config.")
@click.option("--outdir", required=True,
              type=click.Path(file_okay=False),
              help="Output directory for the prediction GeoTIFF.")
@click.option("--batch-size", default=None, type=int, help="Override batch_size.")
def main(train_config, weights, scene, task, outdir, batch_size):
    cfg = E.load_yaml(train_config)
    cfg_dir = os.path.dirname(os.path.abspath(train_config))

    task = task or cfg["task"]
    sim_config_path = E.resolve(cfg_dir, cfg["sim_config"])
    model_config_path = E.resolve(cfg_dir, cfg["model_config"])
    os.makedirs(outdir, exist_ok=True)

    m_cfg = E.load_yaml(model_config_path)
    sim_cfg = E.load_yaml(sim_config_path)
    batch_size = batch_size or cfg["batch_size"]

    min_frac = cfg.get("min_frac")
    if min_frac is None:
        min_frac = sim_cfg["mixing"]["interclass_min_frac"]
    min_frac = float(min_frac)
    classes = list(sim_cfg["classes"])
    n_classes = len(classes)

    # Wavelengths come from the sim config's own drop_wl_ranges application to
    # the scene's raw band grid, guaranteeing the banddef matches the bands
    # the scene is dropped to below.
    from specmix.simulator import MixtureSimulator
    wavelengths = MixtureSimulator(sim_config_path).wavelengths

    print(f"[scene] loading {scene} ...")
    rfl, (n_rows, n_cols) = load_scene(scene, sim_cfg["drop_wl_ranges"])
    assert rfl.shape[1] == len(wavelengths), (
        f"scene has {rfl.shape[1]} good bands, sim has {len(wavelengths)}")
    print(f"[scene] {n_rows}x{n_cols} pixels, {rfl.shape[1]} bands")

    device = get_device()
    print(f"[device] {device}  [weights] {weights}")
    model = E.build_model(m_cfg, wavelengths, n_classes, device)
    E.load_weights(model, weights, device)
    spec = E.make_taskspec(task, n_classes, cfg, min_frac)

    preds = np.zeros((rfl.shape[0], n_classes), dtype=np.float32)
    n_batches = (rfl.shape[0] + batch_size - 1) // batch_size
    for i in tqdm(range(0, rfl.shape[0], batch_size), total=n_batches,
                  desc="Running inference on scene"):
        preds[i:i + batch_size] = E.predict_numpy(
            model, rfl[i:i + batch_size], device, spec, batch_size)
    preds = preds.reshape(n_rows, n_cols, n_classes)

    outpath = os.path.join(
        outdir, f"{os.path.splitext(os.path.basename(scene))[0]}_pred_frac.tif")
    tiff_driver = gdal.GetDriverByName("GTiff")
    opts = ["TILED=YES", "COMPRESS=LZW", "BLOCKXSIZE=256", "BLOCKYSIZE=256"]
    ds = tiff_driver.Create(outpath, n_cols, n_rows, n_classes, gdal.GDT_Float32,
                            options=opts)
    for i, cls in enumerate(classes):
        band = ds.GetRasterBand(i + 1)
        band.WriteArray(preds[:, :, i])
        band.SetNoDataValue(-9999)
        band.SetDescription(cls)
    ds.FlushCache()
    del ds
    print(f"[report] wrote {outpath}")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
