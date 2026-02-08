from typing import Tuple, List, Optional
from collections.abc import Iterator
from torch import Tensor, FloatTensor, LongTensor
from torch.utils.data import DataLoader
from copy import deepcopy

from cover_class.simulation import ForceFracRange, run_simulation, one_hot_encode_simulated_data
from cover_class.dataloader import OrchestratorDataset
from cover_class.simulation import DataArgs, SimulationArgs

class ForcedFractionSimulation(Iterator):
    sim_args: SimulationArgs
    data_args: DataArgs
    num_classes: int
    range_idx: int = -1
    class_idx: int = 0
    latest_simulation_labels: Optional[LongTensor] = None
    one_hot_encode: bool = True

    def __init__(self, 
            dataloader: DataLoader, 
            test_spectra: FloatTensor,
            test_labels: Tensor,
            n_rows: int,
            ranges: List[Tuple[int, int]],
            one_hot_encode: bool = True
        ) -> None:
        ods: OrchestratorDataset = dataloader.dataset # type: ignore
        self.sim_args: SimulationArgs = deepcopy(ods.args.sim_config_args) # type: ignore
        self.sim_args.n_iters = n_rows
        self.data_args = DataArgs(test_spectra, test_labels)

        self.num_classes = ods.args.num_classes
        self.ranges = ranges
        self.one_hot_encode = one_hot_encode
    
    def __next__(self) -> Tuple[FloatTensor, LongTensor]:
        self.range_idx += 1
        if self.range_idx == len(self.ranges):
            self.class_idx += 1
            self.range_idx = 0
        if self.class_idx == self.num_classes:
            raise StopIteration()
                
        simulation_data, simulation_labels, _ = run_simulation(
            self.sim_args, 
            self.data_args, 
            self.data_args.real_spectra.device, 
            self.class_idx, 
            ForceFracRange(*self.ranges[self.range_idx])
        )
        if self.one_hot_encode:
            simulation_labels = one_hot_encode_simulated_data(simulation_labels, self.num_classes) # type: ignore
            self.latest_simulation_labels = simulation_labels
            return simulation_data, simulation_labels
        
        self.latest_simulation_labels = simulation_labels
        return simulation_data, simulation_labels
