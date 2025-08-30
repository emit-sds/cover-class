from typing import *
import torch
from torch import FloatTensor, Tensor, CharTensor
from torch.utils.data import DataLoader
from msgspec import Struct
from pathlib import Path
import math

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


class OrchestratorDataLoader(DataLoader):
    '''
    OrchestratorDataLoader is a non-terminating iterator. Caller must institute it's own break.

    Data Policy:
        - The static and simulated data will be generated/retrieved on a per-batch basis
    '''
    args: OrchestratorDataLoaderArgs
    device: Optional[torch.device] = None
    shuffle = True

    step = 0 
    static_epoch = 0
    static_epoch_step = 0
    static_samples_seen = 0
    is_simulated_batch = False

    _static_idx_order: Optional[CharTensor] = None
    _using_static = False

    def __init__(self, 
                args: OrchestratorDataLoaderArgs, 
                shuffle: bool = True, 
                device: Optional[torch.device] = None
            ) -> None:
        self.args = args; self.shuffle = shuffle; self.device = device
        self._using_static = (self.args.static_labels is not None) and (self.args.static_data is not None)
        if self._using_static: self.__shuffle__()

    def __iter__(self) -> Tuple[torch.FloatTensor, torch.Tensor]: # type: ignore
        ''' This iterator does not stop '''
        self.step += 1
        if self._using_static and self.__static_batch_predicate__():
            self.is_simulated_batch = False
            start =  (self.static_epoch_step * self.args.batch_size)
            end   = ((self.static_epoch_step+1) * self.args.batch_size)
            # mypy doesn't catch self._using_static
            idx = self._static_idx_order[start: end] # type: ignore
            self.static_samples_seen += len(idx)
            if end >= len(self.args.static_data)-1: self.__reset__() # type: ignore
            
            return self.args.static_data[idx], self.args.static_labels[idx] # type: ignore
        
        self.is_simulated_batch = True
        # TODO: need to return simulated data
        return ... # type: ignore
        
    def __static_batch_predicate__(self) -> bool:
        return math.floor((self.static_epoch_step + 1) * self.args.batch_size) != math.floor(self.static_epoch_step * self.args.batch_size)
    
    def __shuffle__(self) -> None:
        if self.shuffle:
            self._static_idx_order = CharTensor(torch.randperm(
                # I don't want to need a check every time here - caller's responsibility
                self.args.static_labels.nelement(), # type: ignore
                dtype=torch.int8, 
                device=self.device
            ))
        
    def __reset__(self) -> None:
        self.static_epoch += 1
        self.static_epoch_step = 0
        self.__shuffle__()
    
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
