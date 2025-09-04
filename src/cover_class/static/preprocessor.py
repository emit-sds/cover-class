from typing import Tuple
from torch import FloatTensor, Tensor
import torch
from numpy.typing import NDArray
import numpy as np
from scipy.interpolate import interp1d # type: ignore[import]

def interior_interpolation(
        data_matrix: NDArray[np.float32], 
        current_wavelengths:NDArray[np.float32],
        ) -> Tuple[FloatTensor, Tensor]:
    wl_min = np.ceil(current_wavelengths.min())
    wl_max = np.floor(current_wavelengths.max())
    target_wavelengths = np.arange(wl_min, wl_max+1, dtype=np.float32)

    f = interp1d(current_wavelengths, data_matrix, kind='linear', axis=1, fill_value="extrapolate")
    interpolated_spectra = f(target_wavelengths)
    return FloatTensor(torch.from_numpy(interpolated_spectra).to(torch.float32)), torch.from_numpy(target_wavelengths)

def convolve(data_matrix:FloatTensor) -> FloatTensor: ... # type: ignore 

