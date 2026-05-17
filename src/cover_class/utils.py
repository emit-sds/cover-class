from typing import Dict, Tuple, List
import yaml # type: ignore[import]
import torch
import numpy as np
import random
from spectral.io import envi # type: ignore[import]
import re
import h5py # type: ignore[import]

def read_config(path: str|Dict) -> Dict: 
    if isinstance(path, dict): return path
    with open(path, 'r') as f: return yaml.safe_load(f)

def seed(s:int):
    # reference: https://docs.pytorch.org/docs/stable/notes/randomness.html
    random.seed(s)
    np.random.seed(s)

    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed") and torch.backends.mps.is_available():
        torch.mps.manual_seed(s)

def name_to_nm(bandname:str) -> float:
    """ Convert wavelength text to float """
    return float(re.search(r'(\d+\.\d+)', bandname).group(1)) # type: ignore 

def load_rfl(hdr_fp:str) -> Tuple[np.ndarray, np.ndarray]:
    rfl_header = envi.open(hdr_fp)
    rfl = rfl_header.open_memmap(interleave='bip')
    banddef = [name_to_nm(name) for name in rfl_header.metadata['wavelength']]
    banddef = np.array(banddef, dtype=float) # type: ignore 
    return rfl, banddef # type: ignore 

def ood_test_set_from_config(c: str|Dict, include_unknown: bool = False, err_on_missed_class: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load the OOD test set from a configuration file.

    Args:
        c: Config file path or a dictionary containing 'datasets' and 'ood-test-set'.
        include_unknown: If True, samples with label 2 (unknown) are included and treated as present.
                        If False, any sample with a label 2 is discarded from the dataset.
        err_on_missed_class: If True, raises a RuntimeError if a class specified in the config is missing from the OOD set.

    Returns:
        A tuple containing:
            - X: A torch.Tensor of spectra.
            - Y: A torch.Tensor of multi-hot labels for the classes in class_order.
    """
    config = read_config(c)
    class_order: List[str] = [d for d in config['datasets'].keys() if config['datasets'][d] is not None]

    with h5py.File(config['ood-test-set'], 'r') as f:
        labels = np.asarray(f['labels'][:])
        spectra = np.asarray(f['spectra'][:])
        classes = np.asarray(f.attrs['classes'][:]).astype(str) # type: ignore

        idx = {c: j for j, c in enumerate(classes)}

        X = torch.from_numpy(spectra).to(torch.float32)
        Y_np = np.zeros((labels.shape[0], len(class_order)), dtype=np.float32)
        for i, name in enumerate(class_order):
            class_labels = labels[:, idx[name]]
            #print(f"Class {name}: 0s: {np.sum(class_labels == 0)}, 1s: {np.sum(class_labels == 1)}, 2s: {np.sum(class_labels == 2)}")
            if include_unknown:
                Y_np[:, i] = np.where(class_labels == 2, 1, class_labels)
            else:
                Y_np[:, i] = np.where(class_labels == 2, np.nan, class_labels)
            if err_on_missed_class and not np.any(Y_np[:, i] == 1):
                raise RuntimeError(f"No data is found in the OOD Test set for class: '{name}'")
        return X, torch.from_numpy(Y_np).to(torch.float32)
