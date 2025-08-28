from typing import Tuple
from torch import FloatTensor, Tensor
from numpy.typing import NDArray
import numpy as np

def interior_interpolation(data_matrix: NDArray[np.float32]) -> Tuple[FloatTensor, Tensor, FloatTensor, Tensor]: ... # type: ignore

def train_test_split(data_matrix: NDArray[np.float32], frac_test: float) -> FloatTensor: ... # type: ignore

def prep_data_from_config(config:str) -> Tuple[FloatTensor, Tensor, FloatTensor, Tensor]: ... # type: ignore
