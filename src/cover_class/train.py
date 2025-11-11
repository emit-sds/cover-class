from pathlib import Path
from typing import Tuple, Dict, Optional
from torch import FloatTensor, Tensor, LongTensor
from torch.utils.data import DataLoader
import torch
import h5py # type: ignore[import]
from copy import deepcopy

from cover_class.dataloader import dataloader_from_config, OrchestratorDataset
from cover_class.utils import read_config, seed as sseed
from cover_class.subsample import subsample_from_config, train_test_split, drop_bad_bands
from cover_class.simulation import run_simulation, SimulationArgs
from cover_class.static.retrieval import make_hdf5

def setup_training_from_config(
        config: str|Dict, 
        batch_size: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
        subsampled_files_outdir: str = ''
    ) -> Tuple[DataLoader, FloatTensor, Tensor]:
    """
    :param: simulated_test_set_n_rows = 0 means don't return a simulated set

    Returns: A tuple of the training dataloader, the test data matrix, and test labels
    """
    if seed is not None:
        sseed(seed)

    train_spectra, train_labels, test_spectra, test_labels = Tensor(), Tensor(), Tensor(), Tensor()

    config = read_config(config)
    drop_bands = config['drop-bands-wavelengths']
    for i, d in enumerate(config['datasets']):
        hdf5_list = config['datasets'][d]
        if hdf5_list is None: continue
        
        # subsampling and train test split will happen on a per file basis
        for hdf5 in hdf5_list:
            with h5py.File(hdf5, 'r') as f:
                file_spectra = f['spectra'][:]
                file_wavelengths = f.attrs['wavelengths']
                file_spectra = drop_bad_bands(file_spectra, file_wavelengths, drop_bands)
                subsampled_spectra = subsample_from_config(config, file_spectra)
                labels = torch.full((subsampled_spectra.shape[0],), i)

                if subsampled_files_outdir != '':
                    method = config['subsample']['selected-method']
                    sconf: dict = config['subsample'][str(method).lower()] if method is not None else dict({})
                    make_hdf5(hdf5, subsampled_files_outdir, d+'_subsampled', file_wavelengths, subsampled_spectra, **sconf)

                X_train, X_test, Y_train, Y_test = train_test_split(subsampled_spectra, labels, config['subsample']['test-fraction'])

                train_spectra = torch.concatenate([train_spectra, X_train], dim=0)
                test_spectra = torch.concatenate([test_spectra, X_test], dim=0)
                train_labels = torch.concatenate([train_labels, Y_train], dim=0)
                test_labels = torch.concatenate([test_labels, Y_test], dim=0)

    train_spectra = train_spectra.to(torch.float32)
    test_spectra = test_spectra.to(torch.float32)

    odl = dataloader_from_config(
        config, 
        FloatTensor(train_spectra),
        LongTensor(train_labels.to(dtype=torch.long)),
        batch_size,
        shuffle,
    )        

    return odl, FloatTensor(test_spectra), test_labels

def make_simulation_test_set(
        odl: DataLoader,
        simulated_test_set_n_rows: int = 0
    ) -> Tuple[FloatTensor, LongTensor, FloatTensor | None]:
    ods: OrchestratorDataset = odl.dataset # type: ignore
    test_sim_config_args: SimulationArgs = deepcopy(ods.args.sim_config_args) # type: ignore
    test_sim_config_args.n_iters = simulated_test_set_n_rows

    return run_simulation(
        test_sim_config_args, 
        ods.args.sim_data_args # type: ignore
    )