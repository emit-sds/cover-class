from typing import *
import torch
from torch import FloatTensor, Tensor, CharTensor
from torch.utils.data import DataLoader
from msgspec import Struct

from cover_class.simulation import run_simulation, SimulationArgs, DataArgs
from cover_class.utils import read_config


class OrchestratorDataLoaderArgs(Struct):
    batch_size: int
    percent_static: float

    sim_config_args: Optional[SimulationArgs]
    sim_data_args: Optional[DataArgs]

    static_data: Optional[torch.FloatTensor]
    static_labels: Optional[torch.Tensor]

    _using_static = False
    _using_sim = False

    # This tells you the ids out of a 100 batches, which ones will be static/simulated
    _method_selection_idxs: CharTensor = CharTensor(torch.zeros(100, dtype=torch.int8))

    def __post_init__(self):
        self._using_static = (self.static_labels is not None) and (self.static_data is not None)
        self._using_sim = (self.sim_config_args is not None) and (self.sim_data_args is not None)
        assert self._using_static or self._using_sim, "Need to provide either simulation or static data arguments"
        if self._using_static and not self._using_sim: self.percent_static = 100.
        elif not self._using_static: self.percent_static = 0.
        assert (self.percent_static is not None) and (0. <= self.percent_static <= 100.), "'percent_static' needs to be between [0, 100]"

        self._method_selection_idxs[:int(self.percent_static)] = 1


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

    def __init__(self, 
                args: OrchestratorDataLoaderArgs, 
                shuffle: bool = True, 
                device: Optional[torch.device] = None
            ) -> None:
        self.args = args; self.shuffle = shuffle; self.device = device
        if self.args._using_static: self.__shuffle__()

    def __next__(self) -> Tuple[torch.FloatTensor, torch.Tensor]:
        ''' This iterator does not stop '''
        self.step += 1
        if self.args._using_static and self.__use_static_predicate__():
            self.is_simulated_batch = False
            start =  (self.static_epoch_step * self.args.batch_size)
            end   = ((self.static_epoch_step+1) * self.args.batch_size)
            # mypy doesn't catch self._using_static
            idx = self._static_idx_order[start: end] # type: ignore
            self.static_samples_seen += len(idx)
            if end >= len(self.args.static_data)-1: # type: ignore
                self.__reset__()

            return self.args.static_data[idx], self.args.static_labels[idx] # type: ignore
        
        elif self.args._using_sim:
            self.is_simulated_batch = True
            return run_simulation(self.args.sim_config_args, self.args.sim_data_args) # type: ignore
        
        else: raise StopIteration()
            
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

    def __use_static_predicate__(self) -> bool:
        return bool(self.args._method_selection_idxs[(self.step % 100)])
    
    def static_epoch_len(self) -> Optional[int]:
        if self.args.static_data is None: return None
        return int(len(self.args.static_data) / self.args.percent_static)

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
