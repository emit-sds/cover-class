from typing import Tuple
from torch import FloatTensor, Tensor

def convex_hull(data_matrix: FloatTensor) -> FloatTensor: ... # type: ignore

def kmeans(data_matrix: FloatTensor) -> FloatTensor: ... # type: ignore

def lhs(data_matrix: FloatTensor) -> FloatTensor: ... # type: ignore

def train_test_split(data_matrix: FloatTensor, frac_test: float) -> FloatTensor: ... # type: ignore

def prep_data_from_config(config:str) -> Tuple[FloatTensor, Tensor, FloatTensor, Tensor]: ... # type: ignore
