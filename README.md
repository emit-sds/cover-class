# Package

## Installation

## Example Usage
### Train Time
```
from cover_class.subsample import prep_data_from_config
from cover_class.dataloader import OrchestratorDataset, dataloader_from_config

my_config_path = '/my/path/config.yaml'
data_train, labels_train, data_test, labels_test = prep_data_from_config(my_config_path)
dataloader: OrchestratorDataset = dataloader_from_config(
    my_config_path, 
    data_train, 
    labels_train,
    batch_size
)
```