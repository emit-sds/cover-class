from typing import Dict
import yaml # type: ignore[import]

def read_config(path: str|Dict) -> Dict: 
    if isinstance(path, dict): return path
    with open(path, 'r') as f: return yaml.safe_load(f)