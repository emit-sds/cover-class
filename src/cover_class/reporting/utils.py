from typing import List, Optional, Any, Union
import torch
from torch import Tensor
import numpy as np
from numpy.typing import NDArray
import h5py #type: ignore[import]

from cover_class.subsample.forward_pipeline import drop_bad_bands

def make_numpy(x: Union[Tensor, NDArray]) -> NDArray:
    return x.detach().cpu().numpy() if isinstance(x, Tensor) else x

def inference_over_scene(
        dataset_fp:str,
        model:Any, 
        drop_wl_ranges:Optional[List[List[int]]] = None,
    ) -> NDArray:
    
    with h5py.File(dataset_fp , "r") as f:
        rfl: NDArray = f['reflectance'][:]
        bands: NDArray = f['sensor_band_parameters']['wavelengths'][:]
    
    original_shape = (rfl.shape[0], rfl.shape[1])
    rfl = rfl.reshape(rfl.shape[0] * rfl.shape[1], rfl.shape[2])

    rfl = drop_bad_bands(rfl, bands, drop_wl_ranges)
    posterior: np.ndarray
    if isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            posterior_t: torch.Tensor = model(torch.from_numpy(rfl).to(dtype=torch.float32))
        posterior = posterior_t.cpu().detach().numpy()
    else:
        posterior = model(rfl)
    posterior = posterior.reshape((original_shape[0], original_shape[1], posterior.shape[-1]))
    return posterior

def rgb_from_scene(dataset_fp:str, red_wl=650, green_wl=560, blue_wl=460) -> NDArray:
    with h5py.File(dataset_fp , "r") as f:
        rfl: NDArray = f['reflectance'][:]
        wavelengths: NDArray = f['sensor_band_parameters']['wavelengths'][:]
    return rfl[..., np.array([np.argmin(abs(wavelengths-w)) for w in [red_wl, green_wl, blue_wl]])]
