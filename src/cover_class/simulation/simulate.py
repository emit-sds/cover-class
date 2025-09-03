from typing import List, Optional, Tuple
import numpy as np
import numpy.typing as npt
from torch import FloatTensor, ByteTensor
import torch

from cover_class.simulation import SimulationArgs, DataArgs


class _ctxblk:
    def __enter__(self, *args): ...
    def __exit__(self, *args): ...
__ctxblk = _ctxblk()


def run_simulation(sim_args: SimulationArgs, data_args: DataArgs) -> Tuple[FloatTensor, ByteTensor]:
    spectra_result: List[npt.NDArray[np.float32]] = []
    label_result:   List[npt.NDArray[np.uint8]] = []
    classes, total_n_components, total_n_comp_idxs = _0_init_simulation_state(sim_args)

    alpha = _1_generate_alpha(sim_args.alpha, sim_args.alpha_uniform_low, sim_args.alpha_uniform_high, total_n_components.sum())

    alpha_idx_start = 0
    for i in range(sim_args.n_iters): # unfortunately, we need to iterate due to how each simulation has different component numbers

        with __ctxblk: # get dirichlet samples
            dirich_fractions, mask = _2_generate_dirichlet_distribution(
                alpha[alpha_idx_start, alpha_idx_start + total_n_components[i]], 
                sim_args.min_frac
            )
            alpha_idx_start += total_n_components[i]
            if (~mask).all(): continue

        with __ctxblk: # remove small fractions and re-weigh the rest
            dirich_fractions = _3_remove_small_fractions(dirich_fractions, mask)

            n_components_iter = np.add.reduceat(mask, np.r_[0, total_n_comp_idxs[i]]) # gets the new number of components per class
            classes_iter      = classes[i][n_components_iter.astype(np.bool_)]

        with __ctxblk: # split
            spectra_subset, label_subset = _4_stratified_split(
                data_args.real_spectra,
                data_args.real_labels,
                classes_iter,
                n_components_iter, 
            )

        with __ctxblk: # add in the noise
            mixed_spectrum = np.sum(spectra_subset.T * dirich_fractions, axis=1, dtype=np.float32)
            mixed_spectrum += _5_add_noise(
                sim_args.noise_covariance,
                total_n_components[i], 
                sim_args.white_noise
            )

        spectra_result.append(mixed_spectrum)
        label_result.append(label_subset)

    spectra = torch.from_numpy(np.concatenate(spectra_result, axis=0)).to(dtype=torch.float32)
    labels = torch.from_numpy(np.concatenate(label_result, axis=0)).to(dtype=torch.uint8)
    return FloatTensor(spectra), ByteTensor(labels)


def _0_init_simulation_state(
        sim_args: SimulationArgs

    ) -> Tuple[
        npt.NDArray[np.uint8],  # classes
        npt.NDArray[np.uint16], # total_n_components
        npt.NDArray[np.uint16]  # total_n_comp_idxs
    ]:
    
    size = (sim_args.n_iters, sim_args.n_classes_in_subsets)
    classes:            npt.NDArray[np.uint8]  = np.argpartition(np.random.random((size[0], sim_args.n_classes)), size[1], axis=1)[:, :size[1]].astype(np.uint8)
    n_components:       npt.NDArray[np.uint16] = np.random.choice(sim_args.n_components, size, replace=True).astype(np.uint16)
    total_n_comp_idxs:  npt.NDArray[np.uint16] = n_components.cumsum(axis=1, dtype=n_components.dtype)
    total_n_components: npt.NDArray[np.uint16] = total_n_comp_idxs[:, -1]
    total_n_comp_idxs = total_n_comp_idxs[:, :-1]
    return classes, total_n_components, total_n_comp_idxs
    

def _1_generate_alpha(
        alpha:              Optional[float],
        alpha_uniform_low:  float,
        alpha_uniform_high: float,
        array_len:          int
        
    ) -> npt.NDArray[np.float16]:
    
    if alpha is not None:
        return np.repeat(alpha, array_len).astype(np.float16)
    else:
        return np.random.uniform(alpha_uniform_low, alpha_uniform_high, array_len).astype(np.float16)
    

def _2_generate_dirichlet_distribution(
        alpha:    npt.NDArray[np.float16],
        min_frac: float
        
    ) -> Tuple[
        npt.NDArray[np.float32], # dirich_fractions
        npt.NDArray[np.bool_]    # mask
    ]:

    # len(dirich_fractions) = (total number of components for this simulation - variable per simulation)
    dirich_fractions: npt.NDArray[np.float32] = np.random.dirichlet(alpha, 1).astype(np.float32)
    mask:             npt.NDArray[np.bool_]   = dirich_fractions >= min_frac
    return dirich_fractions, mask


def _3_remove_small_fractions(
        dirich_fractions: npt.NDArray[np.float32], 
        mask:             npt.NDArray[np.bool_]
        
    ) -> npt.NDArray[np.float32]:

    if not mask.any():
        dirich_fractions[np.argmax(dirich_fractions)] = 1
    survivors = dirich_fractions[mask]
    survivors /= survivors.sum()
    return survivors


def _4_stratified_split(
        real_spectra: npt.NDArray[np.float32], 
        real_labels:  npt.NDArray[np.uint8], 
        classes:      npt.NDArray[np.uint8], 
        n_components: npt.NDArray[np.uint16]

    ) -> Tuple[
        npt.NDArray[np.float32], # spectra_subset
        npt.NDArray[np.uint8]    # label_subset
    ]:

    take_idx = []
    for c, n in zip(classes, n_components):
        idx = np.flatnonzero(real_labels == c)
        take_idx.append(idx[np.random.choice(idx.size, size=n, replace=False)])
    idx = np.concatenate(take_idx)

    return real_spectra[idx], real_labels[idx]


def _5_add_noise(
        sim_args_noise:    Optional[npt.NDArray[np.float32]],
        n_components:      int,
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
