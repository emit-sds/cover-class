# Package

## Installation

## Example Usage
### Train Time
```
from torch.utils.data import DataLoader
from cover_class.subsample import prep_data_from_config
from cover_class.dataloader import OrchestratorDataset, dataloader_from_config

my_config_path = '/my/path/config.yaml'
data_train, labels_train, data_test, labels_test = prep_data_from_config(my_config_path)
dataloader: DataLoader = dataloader_from_config( # the DataLoader has the OrchestratorDataset generator
    my_config_path, 
    data_train, 
    labels_train,
    batch_size,
    shuffle
)
```

### Static Dataset Processing
This is to demonstrate how to create a standardized set of HDF5 datasets for training.
It gets the data matrix from a supported CSV format from either a VFS or downloaded through the network.
```
from cover_class.static.retrieval import generate_hdf5_from_config
generate_hdf5_from_config('/path/to/my/config.yml')
```