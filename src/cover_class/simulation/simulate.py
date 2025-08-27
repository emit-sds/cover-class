from typing import List, Optional, Tuple
from msgspec import Struct
import numpy as np
import numpy.typing as npt
from torch import FloatTensor, ByteTensor

from cover_class.simulation.const import *

class SimulationArgs(Struct):
    n_iters: int
    n_classes: int
    n_classes_in_subsets: int
    min_frac: float
    n_components: List[int] # per class
    alpha: Optional[float]
    noise_std: Optional[npt.NDArray[np.float32]]

class DataArgs(Struct):
    elements: npt.NDArray[np.float32]
    class_targets: npt.NDArray[np.int8]
    wavelengths: Optional[npt.NDArray[np.float16]]

def run_simulation(s: SimulationArgs, d: DataArgs) -> Tuple[FloatTensor, ByteTensor]: ... # type: ignore
