from typing import *
import torch
from torch import FloatTensor, Tensor
from torch.utils.data import Dataset
from msgspec import Struct
from pathlib import Path

from cover_class.simulation import run_simulation, SimulationArgs
from cover_class.dataloader.const import *
from cover_class.utils import read_config


class OrchestratorDataLoaderArgs(Struct):
    batch_size: int
    percent_static: float
    output_path:str
    checkpoint_path:Optional[str]
    static_data:Optional[torch.FloatTensor]
    static_labels:Optional[torch.Tensor]
    sim_args:Optional[SimulationArgs]

    def __post_init__(self):
        Path.is_dir(Path(self.output_path))
        if self.checkpoint_path:
            assert Path.is_dir(Path(self.checkpoint_path))    


class OrchestratorDataLoader(Dataset):
    '''
    Data Policy:
        - The static and simulated data will be generated/retrieved on a per-batch basis
    '''
    args: OrchestratorDataLoaderArgs
    state: Dict = {
        'iter':0,
        'epoch':0,
    }

    def __init__(self, args:OrchestratorDataLoaderArgs, shuffle=False) -> None: ...

    def __get_item__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]: ... # type: ignore
    
    def __len__(self) -> int:
        if self.args.static_data is not None:
            return int(len(self.args.static_data) / self.args.percent_static)
        return MAX_DATALOADER_VALUE


def dataloader_from_config( # type: ignore
        config_path:str, 
        spectra:FloatTensor,
        labels:Tensor,
        batch_size:int,
    ) -> OrchestratorDataLoader: 
    config = read_config(config_path)
    # Dataloader:
    # 1. build the dataloader args
    # 2. build the simulation args
    # 3. build dataloader
    ...
