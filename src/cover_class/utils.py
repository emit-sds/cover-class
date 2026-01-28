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
    config = read_config(c)
    class_order: List[str] = [d for d in config['datasets'].keys() if config['datasets'][d] is not None]

    with h5py.File(config['ood-test-set'], 'r') as f:
        labels  = np.asarray(f['labels'][:])
        classes = np.asarray(f.attrs['classes'][:]).astype(str) # type: ignore

        present = (labels != 0) if include_unknown else (labels != 0) & (labels != 2)
        idx = {c: j for j, c in enumerate(classes)}

        X = torch.from_numpy(f['spectra'][:]).to(torch.float32)
        Y = np.zeros((labels.shape[0], len(class_order)), dtype=np.uint8)
        for i, name in enumerate(class_order):
            Y[:, i] = present[:, idx[name]].astype(np.uint8)
            if err_on_missed_class and Y[:, i].sum() == 0:
                raise RuntimeError(f"No data is found in the OOD Test set for class: '{name}'")
        return X, torch.from_numpy(Y).to(torch.long)