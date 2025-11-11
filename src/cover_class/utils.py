from typing import Dict
import yaml # type: ignore[import]
import torch
import numpy as np
import random

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