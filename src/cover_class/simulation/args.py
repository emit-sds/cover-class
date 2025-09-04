from msgspec import Struct
from typing import List, Optional, Tuple, Dict
import numpy as np
import numpy.typing as npt
from torch import FloatTensor, Tensor

from cover_class.utils import read_config

class SimulationArgs(Struct):
    n_iters: int
    n_classes: int
    n_classes_in_subsets: int
    n_components: List[int] # per class
    min_frac: float
    alpha: Optional[float]
    alpha_uniform_low: float
    alpha_uniform_high: float
    white_noise: float
    noise_covariance: Optional[npt.NDArray[np.float32]]

class DataArgs(Struct):
    real_spectra: npt.NDArray[np.float32]
    real_labels: npt.NDArray[np.uint8]

def args_from_config(config: Dict|str, data_matrix:FloatTensor, labels:Tensor, batch_size:int) -> Tuple[SimulationArgs, DataArgs]:
    config = read_config(config)
    sim_config = config['simulation']
    n_classes = sum(map(bool, config['datasets'].values()))
    noise_cov = None
    if sim_config['noise_covariance_csv']:
        assert str(sim_config['noise_covariance_csv']).endswith('csv'), f'Noise covariance file does not end with .csv: {sim_config['noise_covariance_csv']}'
        noise_cov = np.genfromtxt(sim_config['noise_covariance_csv'], delimiter=',', dtype=float)

    s = SimulationArgs(
        n_iters = batch_size,
        n_classes = n_classes,
        n_classes_in_subsets = sim_config['n_classes_in_subsets'],
        min_frac = sim_config['min_frac'],
        n_components = sim_config['n_components'],
        alpha = sim_config['alpha'],
        alpha_uniform_low = sim_config['alpha_uniform_low'],
        alpha_uniform_high = sim_config['alpha_uniform_high'],
        white_noise = sim_config['white_noise'],
        noise_covariance = noise_cov
    )

    d = DataArgs(
        real_spectra = data_matrix.cpu().numpy(),
        real_labels = labels.cpu().numpy()
    )
    return s, d
