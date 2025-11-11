from typing import Tuple, List
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

def left_edge_scale(
        data_matrix: NDArray[np.float32],
        left_edges: List[int]
    ) -> None:
    """
    This function takes in the data matrix and the left edges of the edge discontinuity. 
    It then scales the left side of the spectra to be on the same magnitude.
    As such, there will be cumulaive scaling of the left-portion of the data matrix until the 
    edges are all sequentially processed/corrected. 
    """
    for edge in sorted(left_edges):
        denom = np.where(data_matrix[:, edge] == 0, 1e-8, data_matrix[:, edge]) # division by 0 protection
        scaling_factors = data_matrix[:, edge+1] / denom
        data_matrix[:, :edge+1] *= scaling_factors[:, np.newaxis]
