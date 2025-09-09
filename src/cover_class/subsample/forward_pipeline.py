from typing import Tuple, Dict
from torch import FloatTensor, Tensor
from numpy.typing import NDArray
import numpy as np
from sklearn.model_selection import train_test_split as tts # type: ignore[import]

from cover_class.utils import read_config
from cover_class.subsample.subsampler import convex_hull, kmeans, kmedoids, lhs

def train_test_split(data_matrix: FloatTensor, labels:Tensor, frac_test: float, seed: int=42) -> Tuple[FloatTensor, FloatTensor, Tensor, Tensor]:
    return tts(data_matrix, labels, test_size=frac_test, random_state=seed)

def subsample_from_config(
        config:str, 
        data_matrix: NDArray[np.float32], 
    ) -> FloatTensor:

    subsample_config:Dict = read_config(config)['subsample']
    method:str = str(subsample_config['selected-method']).lower()
    assert (method_config := subsample_config.get(method, None)) is not None, f'config for {method} not found'
    
    match method:
        case 'convex-hull':
            return convex_hull(data_matrix, **method_config)
        case 'kmeans':
            return kmeans(data_matrix, **method_config)
        case 'kmedoids':
            return kmedoids(data_matrix, **method_config)
        case 'lhs':
            return lhs(data_matrix, **method_config)
        case _:
            raise ValueError(f'Unsupported method: {method}')
    return FloatTensor() # here for mypy