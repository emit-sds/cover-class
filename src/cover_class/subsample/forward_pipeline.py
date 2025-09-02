from typing import Tuple, Any
from torch import FloatTensor, Tensor
from numpy.typing import NDArray
import numpy as np
from sklearn.model_selection import train_test_split as tts # type: ignore[import]

def train_test_split(data_matrix: FloatTensor, labels:Tensor, frac_test: float) -> Tuple[FloatTensor, FloatTensor, Tensor, Tensor]:
    return tts(data_matrix, labels, test_size=frac_test)

def prep_data_from_config(config:str) -> Tuple[FloatTensor, Tensor, FloatTensor, Tensor]: ... # type: ignore
