from typing import Tuple
from pathlib import Path
import pandas as pd # type: ignore[import]
import numpy as np
from numpy.typing import NDArray
import torch
import h5py # type: ignore[import]
from io import BytesIO
import requests # type: ignore[import]

from cover_class.utils import read_config
from cover_class.static.preprocessor import interior_interpolation


def download(uri: str) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    resp = requests.get(
        uri,
        timeout=30,
        verify=True,
        headers={"Accept-Encoding": "gzip, deflate"},
        allow_redirects=True,
    )
    resp.raise_for_status()
    bio = BytesIO(resp.content)
    return vfs_csv(bio)


def vfs_csv(path:str|BytesIO) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    '''
    The CSVs found on the virutal filesystem are expected to be standardized such that:
    - There are a contiguous set of columns whose header values are floating-point wavelength values (in nm)
        and the values of the rows for each of the forementioned columns are the spectral reflectance values.
    '''
    # the largest contiguous set of columns that are floats will be our target
    df = pd.read_csv(path)

    def isfloat(s):
        try:
            float(s) 
            return True
        except ValueError:
            return False

    is_float = np.fromiter(
        map(isfloat, df.columns),
        count=len(df.columns), 
        dtype=bool
    )
    if not is_float.any(): raise ValueError("No wavelength column headers found.")

    starts = np.flatnonzero(is_float & ~np.r_[False, is_float[:-1]])
    ends = np.flatnonzero(is_float & ~np.r_[is_float[1:], False]) + 1
    i = np.argmax(ends - starts)
    s, e = starts[i], ends[i]
    sub = df.iloc[:, s:e]
    
    wavelengths = sub.columns.to_numpy(dtype=np.float32)
    spectra     = sub.to_numpy(dtype=np.float32)
    return wavelengths, spectra


def make_hdf5(original_path:str, outpath:str, class_:str, wavelengths:torch.Tensor, spectra:torch.FloatTensor) -> str:
    '''
    HDF5 contents:
    datasets:
        - 'spectra' [torch.FloatTensor]: the spectra data matrix
    attributes:
        - 'wavelengths' [torch.Tensor]: the wavelength nanometer values for each spectra dimension
        - 'original-ds' [string]: the name of the original dataset this data is from

    HDF5 output location: '{outpath}/{name of class from config file}_{original data filename}.hdf5'
    '''
    ofn = Path(original_path).name
    outname = Path(outpath)/f'{class_}_{ofn}.hdf5'
    with h5py.File(outname, 'w') as f:
        f.create_dataset('spectra', data=spectra)
        f.attrs['wavelengths'] = wavelengths
        f.attrs['original-ds'] = original_path
    return str(outname)


def generate_hdf5_from_config(config_path:str) -> None:
    config = read_config(config_path)
    # get the output directories
    ds = config['datasets']
    outdir = ds['output-directory']
    assert Path(outdir).is_dir(), f"'output-directory': {outdir} is not a directory"

    # get the target wavelengths
    wavelength_file:str = config['target-wavelengths-file']
    assert Path(wavelength_file).exists(), f"'target-wavelengths-file': {wavelength_file} does not exist"
    assert wavelength_file.endswith('.npy'), f"'target-wavelengths-file': {wavelength_file} is not a '.npy' file"
    target_wavelengths: np.ndarray = np.load(wavelength_file)

    for d in (ds_classes := ds['classes']):
        for location in ds_classes[d]:
            # 1. get the wavelength and spectra from the locations
            if Path(location).is_file(): file_wavelengths, spectra = vfs_csv(location)
            else: file_wavelengths, spectra = download(location)

            # 2. interpolate the wavelengths
            spectra_interp = interior_interpolation(spectra, file_wavelengths, target_wavelengths)

            # 3. save hdf5 file
            outname = make_hdf5(
                location, 
                outdir, 
                d, 
                torch.from_numpy(target_wavelengths).to(torch.float32), 
                spectra_interp
            )
            print(f"Wrote file {outname}")

