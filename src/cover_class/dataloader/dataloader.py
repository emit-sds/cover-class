from typing import *
import torch
from torch import FloatTensor, Tensor, CharTensor, LongTensor
from torch.utils.data import IterableDataset, DataLoader
from msgspec import Struct, field

from cover_class.simulation import args_from_config, SimulationArgs, DataArgs
import cover_class.simulation as sim
from cover_class.simulation.simulate import get_fractions_by_class
from cover_class.utils import read_config


class OrchestratorDatasetArgs(Struct):
    batch_size: int
    percent_static: float

    sim_config_args: Optional[SimulationArgs]
    sim_data_args: Optional[DataArgs]

    static_data: Optional[torch.FloatTensor]
    static_labels: Optional[torch.Tensor]

    # Real OOD spectra injected into training as-is (already multi-hot labels, never mixed/simulated).
    ood_data: Optional[torch.FloatTensor] = field(default=None)
    ood_labels: Optional[torch.Tensor] = field(default=None)
    percent_ood: float = field(default=0.0)

    num_classes: int = field(default=0)
    return_fractions: bool = field(default=False)
    _using_static: bool = field(default=False)
    _using_sim: bool   = field(default=False)
    _using_ood: bool   = field(default=False)

    # This tells you the ids out of a 100 batches, which source each will draw from:
    #   0 -> simulated, 1 -> static, 2 -> ood
    _method_selection_idxs: CharTensor = field(default_factory=lambda: CharTensor(torch.zeros(100, dtype=torch.int8)))

    def __post_init__(self):
        self._using_static = (self.static_labels is not None) and (self.static_data is not None)
        self._using_sim = (self.sim_config_args is not None) and (self.sim_data_args is not None)
        self._using_ood = (self.ood_labels is not None) and (self.ood_data is not None)
        assert self._using_static or self._using_sim or self._using_ood, "Need to provide simulation, static, or ood data arguments"

        if not self._using_static: self.percent_static = 0.
        if not self._using_ood: self.percent_ood = 0.

        # When there is no simulation source, the static/ood sources must account for all batches.
        if not self._using_sim:
            total = self.percent_static + self.percent_ood
            assert total > 0, "Need a non-zero percent for static and/or ood data when simulation is unavailable"
            self.percent_static /= total
            self.percent_ood /= total

        assert 0. <= self.percent_static <= 1.00, "'percent_static' needs to be between [0, 1.]"
        assert 0. <= self.percent_ood <= 1.00, "'percent_ood' needs to be between [0, 1.]"
        assert self.percent_static + self.percent_ood <= 1.0 + 1e-6, "'percent_static' + 'percent_ood' must be <= 1."

        if self._using_sim:
            self.num_classes = self.sim_config_args.n_classes
        elif self._using_static:
            self.num_classes = len(torch.unique(self.static_labels))
        else:
            self.num_classes = self.ood_labels.shape[1]

        n_static = int(self.percent_static * 100)
        n_ood = int(self.percent_ood * 100)
        self._method_selection_idxs[:n_static] = 1
        self._method_selection_idxs[n_static:n_static + n_ood] = 2
        if not self._using_sim:
            # Assign any slots left unallocated by rounding to an available static/ood source.
            remaining = self._method_selection_idxs == 0
            self._method_selection_idxs[remaining] = 1 if self._using_static else 2
        self._shuffle_method_selection_idxs()

    def _shuffle_method_selection_idxs(self):
        self._method_selection_idxs = self._method_selection_idxs[torch.randperm(self._method_selection_idxs.size(0))]


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
    ood_epoch = 0
    ood_epoch_step = 0
    ood_samples_seen = 0
    is_simulated_batch = False
    batch_dirichlet_fraction_store: Optional[FloatTensor] = None

    _static_idx_order: Optional[LongTensor] = None
    _ood_idx_order: Optional[LongTensor] = None

    def __init__(self,
                args: OrchestratorDatasetArgs,
                shuffle: bool = True,
            ) -> None:
        self.args = args; self.shuffle = shuffle
        if self.args._using_static: self.__shuffle__()
        if self.args._using_ood: self.__shuffle_ood__()

    def __iter__(self) -> Iterator[Tuple[torch.FloatTensor, torch.Tensor]]:
        ''' This iterator does not stop '''
        def make_one_hot(y:torch.Tensor) -> torch.Tensor:
            if self.is_simulated_batch:
                # 2D one-hot
                return sim.one_hot_encode_simulated_data(y, self.args.num_classes)
            return torch.nn.functional.one_hot(y, num_classes=self.args.num_classes).double()

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
                if end >= len(self.args.static_data)-1: # type: ignore
                    self.__reset__()

                if self.args.return_fractions:
                    # For static data, create one-hot-like fractions (1.0 for true class, 0.0 for others)
                    labels = make_one_hot(self.args.static_labels[idx]).to(dtype=torch.float32) # type: ignore
                else:
                    labels = make_one_hot(self.args.static_labels[idx]) # type: ignore
                self.batch_dirichlet_fraction_store = None
                yield self.args.static_data[idx], labels # type: ignore

            elif self.args._using_ood and self.__use_ood_predicate__():
                # Real OOD spectra provided as-is: labels are already multi-hot and are
                # NEVER mixed or simulated. Unknown entries carry a -1 sentinel to be masked.
                self.is_simulated_batch = False
                start =  (self.ood_epoch_step * self.args.batch_size)
                end   = ((self.ood_epoch_step+1) * self.args.batch_size)
                self.ood_epoch_step += 1
                idx = self._ood_idx_order[start: end] # type: ignore
                self.ood_samples_seen += len(idx)
                if end >= len(self.args.ood_data)-1: # type: ignore
                    self.__reset_ood__()
                self.batch_dirichlet_fraction_store = None
                yield self.args.ood_data[idx], self.args.ood_labels[idx].to(dtype=torch.float32) # type: ignore

            elif self.args._using_sim:
                self.is_simulated_batch = True
                # mypy doesn't catch self.args._using_sim
                data, labels, fractions = sim.run_simulation(self.args.sim_config_args, self.args.sim_data_args) # type: ignore
                self.batch_dirichlet_fraction_store = fractions

                if self.args.return_fractions:
                    # Return fractions directly (already in correct format from run_simulation)
                    yield data, fractions
                else:
                    yield data, make_one_hot(labels)

            else: raise StopIteration()

    def __shuffle__(self) -> None:
        if self.shuffle:
            self._static_idx_order = LongTensor(torch.randperm(
                self.args.static_labels.size(0), # type: ignore # caller's responsibility
                device=self.args.static_labels.device, # type: ignore
                dtype=torch.int64,
            ))
            self.args._shuffle_method_selection_idxs()

    def __shuffle_ood__(self) -> None:
        if self.shuffle:
            self._ood_idx_order = LongTensor(torch.randperm(
                self.args.ood_labels.size(0), # type: ignore # caller's responsibility
                device=self.args.ood_labels.device, # type: ignore
                dtype=torch.int64,
            ))
        elif self._ood_idx_order is None:
            self._ood_idx_order = LongTensor(torch.arange(self.args.ood_labels.size(0), dtype=torch.int64)) # type: ignore

    def __reset__(self) -> None:
        self.static_epoch += 1
        self.static_epoch_step = 0
        self.__shuffle__()

    def __reset_ood__(self) -> None:
        self.ood_epoch += 1
        self.ood_epoch_step = 0
        self.__shuffle_ood__()

    def __use_static_predicate__(self) -> bool:
        return self.args._method_selection_idxs[(self.step % 100)].item() == 1

    def __use_ood_predicate__(self) -> bool:
        return self.args._method_selection_idxs[(self.step % 100)].item() == 2


def dataloader_from_config(
        config: Dict|str,
        spectra:FloatTensor,
        labels:LongTensor,
        batch_size:int,
        shuffle: bool = True,
        return_fractions: bool = False,
        misc_dataloader_params: dict = {},
        ood_spectra: Optional[FloatTensor] = None,
        ood_labels: Optional[Tensor] = None,
    ) -> DataLoader:

    config = read_config(config)
    sim_config_args, sim_data_args = args_from_config(config, spectra, labels.long(), batch_size)

    ods_args = OrchestratorDatasetArgs(
        batch_size,
        config["dataloader"]["percent-static-data"],
        sim_config_args,
        sim_data_args,
        spectra,
        labels.long(),
        ood_data=ood_spectra,
        ood_labels=ood_labels,
        percent_ood=config["dataloader"].get("percent-ood-data", 0.0),
        return_fractions=return_fractions,
    )
    ods = OrchestratorDataset(ods_args, shuffle)
    return DataLoader(ods, batch_size=None, **misc_dataloader_params)
