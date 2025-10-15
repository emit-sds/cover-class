from typing import Tuple, Dict
from torch import FloatTensor, Tensor
from torch.utils.data import DataLoader
import torch
import h5py # type: ignore[import]

from cover_class.dataloader import dataloader_from_config
from cover_class.utils import read_config
from cover_class.subsample import subsample_from_config, train_test_split, drop_bad_bands

def setup_training_from_config(
        config: str|Dict, 
        batch_size: int,
        shuffle = True,
    ) -> Tuple[DataLoader, FloatTensor, Tensor]:
    """
    Returns: A tuple of the training dataloader, the test data matrix, and test labels
    """

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
        train_labels,
        batch_size,
        shuffle,
    )

    return odl, FloatTensor(test_spectra), test_labels