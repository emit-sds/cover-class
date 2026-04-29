import os

# Set CUBLAS_WORKSPACE_CONFIG for deterministic CUDA behavior
# pylint: disable=wrong-import-position
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import rich_click as click
import yaml

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from osgeo import gdal
from tqdm import tqdm

from cover_class.utils import seed as sseed #type: ignore
from cover_class.utils import read_config
from cover_class.subsample.forward_pipeline import drop_bad_bands, drop_bad_banddef

from spectf.model import SpecTfEncoder
from spectf.utils import get_device

ENV_VAR_PREFIX = 'COVER_CLASS_SCENE_'


class RasterDataset(Dataset):
    def __init__(self, nc_path, config_path):
        super().__init__()

        self.nc_path = nc_path
        self.config_path = config_path

        self.data_config = read_config(config_path)
        self.drop_wl_ranges = self.data_config["drop-bands-wavelengths"]

        with h5py.File(nc_path , "r") as f:
            rfl = np.array(f['reflectance'][:])
            bands = np.array(f['sensor_band_parameters']['wavelengths'][:])

        self.orig_shape = (rfl.shape[0], rfl.shape[1])
        rfl = rfl.reshape(rfl.shape[0] * rfl.shape[1], rfl.shape[2])
        self.rfl = drop_bad_bands(rfl, bands, self.drop_wl_ranges)
        self.banddef = drop_bad_banddef(bands, self.drop_wl_ranges)

    def __len__(self):
        return self.rfl.shape[0]
    
    def __getitem__(self, idx):
        return self.rfl[idx]

@click.command()
@click.option(
    "--outdir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Output file directory.",
    envvar=f'{ENV_VAR_PREFIX}OUTDIR'
)
@click.option(
    "--data-config",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the YAML config for the dataloader.",
    envvar=f'{ENV_VAR_PREFIX}_DATA_CONFIG'
)
@click.option(
    "--model-config",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the YAML config for the model architecture.",
    envvar=f'{ENV_VAR_PREFIX}_MODEL_CONFIG'
)
@click.option(
    "--model-weights",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the model weights file (.pth).",
    envvar=f'{ENV_VAR_PREFIX}_MODEL_WEIGHTS'
)
@click.option(
    "--scene",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Filepaths to the scene to run inference on.",
)
def run_scene(
        outdir: str,
        data_config: str,
        model_config: str,
        model_weights: str,
        scene: str
    ):

    # Load model config
    with open(model_config, 'r', encoding='utf-8') as f:
        m_config = yaml.safe_load(f)
    # batch size
    bs = m_config['batch_size']

    # Create the dataset
    sseed(m_config['random_seed'])
    dataset = RasterDataset(scene, data_config)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)
    banddef = torch.tensor(dataset.banddef)

    # hardcoded GPU 0
    device = get_device(0)
    ds_conf = dataset.data_config['datasets']
    class_names = [str(c) for c in ds_conf.keys() if ds_conf[c] is not None and len(ds_conf[c])]

    # model definition
    model = SpecTfEncoder(banddef.to(dtype=torch.float32, device=device),
                         dim_output=m_config['model']['dim_output'],
                         num_heads=m_config['model']['num_heads'],
                         dim_proj=m_config['model']['dim_proj'],
                         dim_ff=m_config['model']['dim_ff'],
                         dropout=m_config['model']['dropout'],
                         agg=m_config['model']['agg'],
                         use_residual=m_config['model']['use_residual'],
                         num_layers=m_config['model']['num_layers']).to(device)

    # Load model weights
    print(f"Loading model weights from {model_weights}")
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.eval()

    # Inference loop
    predicted_fractions = np.zeros((len(dataset), m_config['model']['dim_output']), dtype=np.float32)
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Running inference on scene"):
        batch = batch.to(device=device, dtype=torch.float32)
        batch = torch.unsqueeze(batch, -1)
        with torch.no_grad():
            logits = model(batch)
            batch_y_hat = torch.sigmoid(logits)
            batch_y_hat = batch_y_hat.detach().cpu().numpy().astype(np.float32)

        predicted_fractions[i*bs:i*bs+batch.shape[0]] = batch_y_hat
    predicted_fraction = predicted_fractions.reshape(dataset.orig_shape[0], dataset.orig_shape[1], m_config['model']['dim_output'])

    outpath = os.path.join(outdir, f"{os.path.basename(scene).split('.')[0]}_pred_frac.tif")

    mem_driver, tiff_driver = gdal.GetDriverByName('MEM'), gdal.GetDriverByName('GTiff')
    opts = ['TILED=YES', 'COMPRESS=LZW', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'COPY_SRC_OVERVIEWS=YES']
    ds = mem_driver.Create('', dataset.orig_shape[1], dataset.orig_shape[0], m_config['model']['dim_output'], gdal.GDT_Float32)
    for i in range(m_config['model']['dim_output']):
        ds.GetRasterBand(i+1).WriteArray(predicted_fraction[:,:,i])
        ds.GetRasterBand(i+1).SetNoDataValue(-9999)
        ds.GetRasterBand(i+1).SetDescription(class_names[i] if i < len(class_names) else f'class_{i}')
    _ = tiff_driver.CreateCopy(outpath, ds, options=opts)
    del _
    del ds


if __name__ == "__main__": 
    # pylint: disable=no-value-for-parameter
    run_scene()
