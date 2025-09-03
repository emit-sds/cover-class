from typing import *
import torch
from torch import FloatTensor, Tensor, CharTensor, LongTensor
from torch.utils.data import IterableDataset, DataLoader
from msgspec import Struct, field

from cover_class.simulation import args_from_config, SimulationArgs, DataArgs
import cover_class.simulation as sim
from cover_class.utils import read_config


class OrchestratorDatasetArgs(Struct):
    batch_size: int
    percent_static: float

    sim_config_args: Optional[SimulationArgs]
    sim_data_args: Optional[DataArgs]

    static_data: Optional[torch.FloatTensor]
    static_labels: Optional[torch.Tensor]

    _using_static: bool = field(default=False)
    _using_sim: bool   = field(default=False)

    # This tells you the ids out of a 100 batches, which ones will be static/simulated
    _method_selection_idxs: CharTensor = field(default_factory=lambda: CharTensor(torch.zeros(100, dtype=torch.int8)))

    def __post_init__(self):
        self._using_static = (self.static_labels is not None) and (self.static_data is not None)
        self._using_sim = (self.sim_config_args is not None) and (self.sim_data_args is not None)
        assert self._using_static or self._using_sim, "Need to provide either simulation or static data arguments"
        if self._using_static and not self._using_sim: self.percent_static = 100.
        elif not self._using_static: self.percent_static = 0.
        assert (self.percent_static is not None) and (0. <= self.percent_static <= 100.), "'percent_static' needs to be between [0, 100]"

        self._method_selection_idxs[:int(self.percent_static)] = 1


class OrchestratorDataset(IterableDataset):
    '''
    OrchestratorDataset is a non-terminating iterator. Caller must institute it's own break.

    Data Policy:
        - The static and simulated data will be generated/retrieved on a per-batch basis

    Example use:
    >>> old = OrchestratorDataset(args)
    >>> dl = DataLoader(old, batch_size=None)
    >>> # NOTE: The `batch_size` in the dataloader must be set to `None`
    >>> for X, Y in dl:
    >>>     ...
    '''
    args: OrchestratorDatasetArgs
    shuffle = True

    step = 0
    static_epoch = 0
    static_epoch_step = 0
    static_samples_seen = 0
    is_simulated_batch = False

    _static_idx_order: Optional[LongTensor] = None

    def __init__(self, 
                args: OrchestratorDatasetArgs, 
                shuffle: bool = True, 
            ) -> None:
        self.args = args; self.shuffle = shuffle
        if self.args._using_static: self.__shuffle__()

    def __iter__(self) -> Iterator[Tuple[torch.FloatTensor, torch.Tensor]]:
        ''' This iterator does not stop '''
        while True:
            self.step += 1
            if self.args._using_static and self.__use_static_predicate__():
                self.is_simulated_batch = False
                start =  (self.static_epoch_step * self.args.batch_size)
                end   = ((self.static_epoch_step+1) * self.args.batch_size)
                self.static_epoch_step += 1
                # mypy doesn't catch self.args._using_static
                idx = self._static_idx_order[start: end] # type: ignore
                self.static_samples_seen += len(idx)
                if end >= len(self.args.static_data): # type: ignore
                    self.__reset__()
                yield self.args.static_data[idx], self.args.static_labels[idx] # type: ignore

            elif self.args._using_sim:
                self.is_simulated_batch = True
                # mypy doesn't catch self.args._using_sim
                yield sim.run_simulation(self.args.sim_config_args, self.args.sim_data_args) # type: ignore

            else: raise StopIteration()
            
    def __shuffle__(self) -> None:
        if self.shuffle:
            self._static_idx_order = LongTensor(torch.randperm(
                self.args.static_labels.nelement(), # type: ignore # caller's responsibility
                device=self.args.static_labels.device, # type: ignore
                dtype=torch.int64, 
            ))
        
    def __reset__(self) -> None:
        self.static_epoch += 1
        self.static_epoch_step = 0
        self.__shuffle__()

    def __use_static_predicate__(self) -> bool:
        return bool(self.args._method_selection_idxs[(self.step % 100)])


def dataloader_from_config(
        config: Dict|str, 
        spectra:FloatTensor,
        labels:Tensor,
        batch_size:int,
        shuffle: bool = True, 
    ) -> DataLoader:

    config = read_config(config)
    sim_config_args, sim_data_args = args_from_config(config, batch_size)

    ods_args = OrchestratorDatasetArgs(
        batch_size,
        config["dataloader"]["percent-static-data"],
        sim_config_args,
        sim_data_args,
        spectra,
        labels
    )
    ods = OrchestratorDataset(ods_args, shuffle)
    return DataLoader(ods, batch_size=None)