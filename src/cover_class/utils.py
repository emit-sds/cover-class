from typing import Dict
import yaml

def read_config(path:str) -> Dict: 
    with open(path, 'r') as f: return yaml.safe_load(f)