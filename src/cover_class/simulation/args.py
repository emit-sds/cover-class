from msgspec import Struct
from typing import List, Optional, Tuple, Dict
import numpy as np
from torch import FloatTensor, Tensor
import torch
from dataclasses import dataclass

from cover_class.utils import read_config

class SimulationArgs(Struct):
    n_iters: int
    n_classes: int
    n_classes_in_subsets: int
    n_components: List[List[int]]
    min_frac: float
    alpha: Optional[float]
    alpha_uniform_low: float
    alpha_uniform_high: float
    white_noise: float
    noise_scalar: Optional[float]
    noise_covariance: Optional[FloatTensor]
    return_fractions: bool
    glint_scalar_range: Tuple[Optional[float], Optional[float]]
    water_classes: List[int]
    class_names: List[str]
    forced_fractions: Dict[str, List[Tuple[float,float]]]

    def __post_init__(self):
        assert len(self.n_components) == self.n_classes, f"Number of classes, {self.n_classes}, must match the number of component sampling ranges, {len(self.n_components)}"
        assert all(len(i) for i in self.n_components), "All classes must have a non-empty value for the `n_components` config"
        assert isinstance(self.forced_fractions, dict), "simulation/forced_fraction_test_set needs to contain key-value pairs"

    def to(self, device: torch.device):
        if self.noise_covariance is not None: 
            self.noise_covariance = self.noise_covariance.to(device) # type: ignore

class DataArgs(Struct):
    real_spectra: FloatTensor
    real_labels: Tensor

    def __post_init__(self):
        assert self.real_spectra.shape[0] == self.real_labels.shape[0], "Spectra and Labels need to have the same number of rows."

    def to(self, device: torch.device):
        self.real_spectra = self.real_spectra.to(device) # type: ignore
        self.real_labels = self.real_labels.to(device)

def args_from_config(config: Dict|str, data_matrix:FloatTensor, labels:Tensor, batch_size:int) -> Tuple[SimulationArgs, DataArgs]:
    config = read_config(config)
    sim_config:dict = config['simulation']
    n_classes = sum(map(bool, config['datasets'].values()))
    noise_cov = None
    if sim_config['noise_covariance_csv']:
        assert str(sim_config['noise_covariance_csv']).endswith('csv'), f"Noise covariance file does not end with .csv: {sim_config['noise_covariance_csv']}"
        noise_cov = np.genfromtxt(sim_config['noise_covariance_csv'], delimiter=',', dtype=float)

    n_components: List[List[int]] = [
        sim_config['n_components'][dataset_name]
        for dataset_name, enabled in config['datasets'].items()
        if enabled and dataset_name in sim_config['n_components']
    ]

    class_names = [c for c,v in config['datasets'].items() if v is not None]
    ffracs = sim_config.get("forced_fraction_test_set", {})

    s = SimulationArgs(
        n_iters = batch_size,
        n_classes = n_classes,
        n_classes_in_subsets = sim_config['n_classes_in_subsets'],
        min_frac = sim_config['min_frac'],
        n_components = n_components,
        alpha = sim_config['alpha'],
        alpha_uniform_low = sim_config['alpha_uniform_low'],
        alpha_uniform_high = sim_config['alpha_uniform_high'],
        white_noise = sim_config['white_noise'],
        noise_scalar = sim_config['noise_scalar'],
        noise_covariance = FloatTensor(torch.from_numpy(noise_cov).to(torch.float32)) if noise_cov is not None else None,
        return_fractions = sim_config["return_fractions"],
        glint_scalar_range = (sim_config["glint_lower_scalar"], sim_config["glint_upper_scalar"]),
        water_classes = [i for i in range(len(config['datasets'])) if 'water' in list(config['datasets'].keys())[i].lower()],
        class_names = class_names,
        forced_fractions = ffracs,
    )

    d = DataArgs(
        real_spectra = data_matrix,
        real_labels = labels
    )
    return s, d

@dataclass
class ForceFracRange:
    low: float
    high: float
