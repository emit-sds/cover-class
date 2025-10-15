from typing import Tuple, Any, Optional, List
import torch
from torch import FloatTensor, Tensor
from numpy.typing import NDArray
import numpy as np
from sklearn.model_selection import train_test_split as tts # type: ignore[import]

def train_test_split(data_matrix: FloatTensor, labels:Tensor, frac_test: float, seed: int=42) -> Tuple[FloatTensor, FloatTensor, Tensor, Tensor]:
    return tts(data_matrix, labels, test_size=frac_test, random_state=seed)

def subsample_from_config(config:str) -> Tuple[FloatTensor, Tensor, FloatTensor, Tensor]: ... # type: ignore

def drop_bad_bands(
        data_matrix: FloatTensor,
        banddef: Tensor, 
        drop_wl_ranges: Optional[List[List[int]]] = None,
    ) -> FloatTensor:
    """
    References https://github.com/emit-sds/SpecTf/blob/main/spectf/utils.py#L69
    Removes bands/wavelengths of high uncertainty from a single spectra
    or an array of spectras.
    """
    if drop_wl_ranges is None or not len(drop_wl_ranges):
        return data_matrix
    
    mask = torch.ones_like(banddef, dtype=torch.bool)
    for low, high in drop_wl_ranges:
        mask ^= (banddef >= low) & (banddef <= high)
    return FloatTensor(data_matrix[..., mask])
