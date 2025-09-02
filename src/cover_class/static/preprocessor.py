from torch import FloatTensor
import torch
from numpy.typing import NDArray
import numpy as np
from scipy.interpolate import interp1d # type: ignore[import]

def interior_interpolation(
        data_matrix: NDArray[np.float32], 
        current_wavelengths:NDArray[np.float32],
        target_wavelengths:NDArray[np.float32],
        ) -> FloatTensor:
    f = interp1d(current_wavelengths, data_matrix, kind='linear', axis=1, fill_value="extrapolate")
    interpolated_spectra = f(target_wavelengths)
    return FloatTensor(torch.from_numpy(interpolated_spectra).to(torch.float32))

def convolve(data_matrix:FloatTensor) -> FloatTensor: ... # type: ignore 

