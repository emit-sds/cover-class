from typing import Tuple, Dict, Optional, List
import torch
import torch
from torch import FloatTensor, Tensor
from numpy.typing import NDArray
import numpy as np
from sklearn.model_selection import train_test_split as tts # type: ignore[import]

from cover_class.utils import read_config
from cover_class.subsample.subsampler import convex_hull, kmeans, kmedoids, lhs

def train_test_split(data_matrix: FloatTensor, labels:Tensor, frac_test: float, seed: int=42) -> Tuple[FloatTensor, FloatTensor, Tensor, Tensor]:
    return tts(data_matrix, labels, test_size=frac_test, random_state=seed)

def subsample_from_config(
        config:str|Dict, 
        file_name: str,
        data_matrix: NDArray[np.float32], 
    ) -> Tuple[FloatTensor, Optional[str], Optional[dict]]:

    subsample_config:Dict = read_config(config)['subsample']
    specific = subsample_config.get('file-specific', None)
    if specific is not None and file_name in specific:
        method:str  = str(list(specific[file_name].keys())[0]).lower()
        params:dict = specific[file_name][method]
    else:
        method = str(subsample_config['selected-method']).lower()
        params = subsample_config.get(method, None)
    
    match method:
        case 'convex-hull':
            return convex_hull(data_matrix, **params), method, params
        case 'kmeans':
            return kmeans(data_matrix, **params), method, params
        case 'kmedoids':
            return kmedoids(data_matrix, **params), method, params
        case 'lhs':
            return lhs(data_matrix, **params), method, params
        case _:
            return FloatTensor(torch.from_numpy(data_matrix).to(torch.float32)), None, params
    return FloatTensor() # here for mypy

def drop_bad_bands(
        data_matrix: NDArray[np.float32],
        banddef: NDArray, 
        drop_wl_ranges: Optional[List[List[int]]] = None,
    ) -> NDArray[np.float32]:
    """
    References https://github.com/emit-sds/SpecTf/blob/main/spectf/utils.py#L69
    Removes bands/wavelengths of high uncertainty from a single spectra
    or an array of spectras.
    """
    if drop_wl_ranges is None or not len(drop_wl_ranges):
        return data_matrix
    
    mask = np.ones_like(banddef, dtype=bool)
    for low, high in drop_wl_ranges:
        mask ^= (banddef >= low) & (banddef <= high)
    spectra = np.delete(data_matrix, ~mask, axis=-1)
    return spectra

def drop_bad_banddef(
        banddef: NDArray, 
        drop_wl_ranges: Optional[List[List[int]]] = None,
    ) -> NDArray[np.float32]:
    """
    References https://github.com/emit-sds/SpecTf/blob/main/spectf/utils.py#L104
    Removes bands/wavelengths of high uncertainty from band definition.
    """
    if drop_wl_ranges is None or not len(drop_wl_ranges):
        return banddef 

    mask = np.ones_like(banddef, dtype=bool)
    for low, high in drop_wl_ranges:
        mask ^= (banddef >= low) & (banddef <= high)
    banddef = banddef[mask]

    return banddef