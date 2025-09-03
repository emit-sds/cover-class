from typing import List, Optional, Tuple, Dict
from msgspec import Struct
import numpy as np
import numpy.typing as npt
from torch import FloatTensor, ByteTensor, Tensor
import torch

from cover_class.utils import read_config

class _ctxblk:
    def __enter__(self, *args): ...
    def __exit__(self, *args): ...
__ctxblk = _ctxblk()

class SimulationArgs(Struct):
    n_iters: int
    n_classes: int
    n_classes_in_subsets: int
    min_frac: float
    n_components: List[int] # per class
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

def run_simulation(sim_args: SimulationArgs, data_args: DataArgs) -> Tuple[FloatTensor, ByteTensor]:
    size = (sim_args.n_iters, sim_args.n_classes_in_subsets)
    spectra_result: List[npt.NDArray[np.float32]] = []
    label_result: List[npt.NDArray[np.uint8]] = []
    classes:            npt.NDArray[np.int8]  = np.argpartition(np.random.random((size[0], sim_args.n_classes)), size[1], axis=1)[:, :size[1]].astype(np.int8)
    n_components:       npt.NDArray[np.int16] = np.random.choice(sim_args.n_components, size, replace=True).astype(np.int16)
    total_n_comp_idxs:  npt.NDArray[np.int16] = n_components.cumsum(axis=1, dtype=n_components.dtype)
    total_n_components: npt.NDArray[np.int16] = total_n_comp_idxs[:, -1]
    total_n_comp_idxs = total_n_comp_idxs[:, :-1]
    total_arr_size: int = total_n_components.sum()

    # create the alpha params
    if sim_args.alpha is not None:
        alpha: npt.NDArray[np.float16] = np.repeat(sim_args.alpha, total_arr_size)
    else:
        alpha = np.random.uniform(sim_args.alpha_uniform_low, sim_args.alpha_uniform_high, total_arr_size).astype(np.float16)

    alpha_idx_start = 0
    for i in range(sim_args.n_iters): # unfortunately, need neet to iterate due to how each simulation has different component numbers

        with __ctxblk: # get dirichlet samples
            alpha_idx_end = alpha_idx_start + total_n_components[i]
            # len(dirich_fractions) = (total number of components for this simulation)
            dirich_fractions: npt.NDArray[np.float32] = np.random.dirichlet(alpha[alpha_idx_start:alpha_idx_end], 1).astype(np.float32)
            mask:             npt.NDArray[np.bool_]   = dirich_fractions >= sim_args.min_frac
            alpha_idx_start += total_n_components[i]
            if (~mask).all(): continue

        with __ctxblk: # remove small fractions and re-weigh the rest
            if not mask.any():
                dirich_fractions[np.argmax(dirich_fractions)] = 1
            survivors = dirich_fractions[mask]
            survivors /= survivors.sum()

            n_components_iter = np.add.reduceat(mask, np.r_[0, total_n_comp_idxs[i]]) # gets the new number of components per class
            classes_iter      = classes[i][n_components_iter.astype(np.bool_)]

        with __ctxblk: # split
            spectra_subset, label_subset = stratified_split(
                data_args.real_spectra,
                data_args.real_labels,
                classes_iter,
                n_components_iter, 
            )

        with __ctxblk: # add in the noise
            mixed_spectrum = np.sum(spectra_subset.T * dirich_fractions, axis=1, dtype=np.float32)
            mixed_spectrum += add_noise(sim_args.noise_covariance, total_n_components[i], sim_args.white_noise)

        spectra_result.append(mixed_spectrum)
        label_result.append(label_subset)

    spectra = torch.from_numpy(np.concatenate(spectra_result, axis=0)).to(dtype=torch.float32)
    labels = torch.from_numpy(np.concatenate(label_result, axis=0)).to(dtype=torch.uint8)
    return FloatTensor(spectra), ByteTensor(labels)
    

def stratified_split(
        real_spectra: npt.NDArray[np.float32], 
        real_labels: npt.NDArray[np.uint8], 
        classes: npt.NDArray[np.int8], 
        sizes: npt.NDArray[np.int16]
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:

    take_idx = []
    for c, n in zip(classes, sizes):
        idx = np.flatnonzero(real_labels == c)
        take_idx.append(idx[np.random.choice(idx.size, size=n, replace=False)])
    idx = np.concatenate(take_idx)

    return real_spectra[idx], real_labels[idx]


def add_noise(
        sim_args_noise:Optional[npt.NDArray[np.float32]],
        n_components: int,
        white_noise_scale: float,
    ) -> npt.NDArray[np.float32]:
    

    if sim_args_noise is not None:
        if len(sim_args_noise.shape) < 2:
            # ---- option 1: scale diagonal std
            
            weights = np.diag(sim_args_noise)
            scale = np.random.normal(loc = 1)
            means = np.zeros(weights.shape[0])
            noise = np.random.multivariate_normal(means, weights * np.abs(scale), 1).ravel()
            noise = noise * np.sign(scale)
        
        else:
            # ---- option 2: use full smooth cov matrix
            means = np.zeros(sim_args_noise.shape[0])
            noise = np.random.multivariate_normal(means, sim_args_noise, 1).ravel()
        
        # add white noise
        white_noise =  np.random.normal(loc = means, scale = white_noise_scale)
        noise = noise + white_noise
        
    else:
        scale = 1
        noise = np.repeat(0, n_components).astype(np.float32)

    return noise.astype(np.float32)

