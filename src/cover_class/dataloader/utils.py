from typing import Dict, Tuple
import torch
from torch import FloatTensor, Tensor
import h5py # type: ignore[import]

from cover_class.utils import read_config

def hdf5_file_handler(config: Dict|str) -> Tuple[FloatTensor, Tensor]:
    config = read_config(config)
    datasets: Dict = config['datasets']
    spectra = torch.empty(0, dtype=torch.float32)
    labels = torch.empty(0)
    for i, paths in enumerate(datasets.values()):
        for p in paths:
            with h5py.File(p, 'r') as f:
                s = f['spectra'][:]
                spectra = torch.cat((spectra, s), dim=0)
                labels = torch.cat((labels, torch.full((len(s),), i)), dim=0)
    return FloatTensor(spectra), labels