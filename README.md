# Package

## Installation

## Example Usage
### Train Time
```
from cover_class.train import setup_training_from_config

dataloader, X_test, Y_test = setup_training_from_config(
    '/my/path/config.yaml',
    batch_size,
    shuffle = True,
)
```

To get the dirichlet fractions, after every iteration, check `dataloader.dataset.batch_dirichlet_fraction_store` for the fractions. (Assuming that `simulation.return_fractions` was set to `true` in dataloader.yml)

To generate a simulated test set:
```
from cover_class.train import test_data, test_labels, test_fractions = make_simulation_test_set(
    dataloader,
    simulated_test_set_n_rows = <some_number>
)
```

### Static Dataset Processing
This is to demonstrate how to create a standardized set of HDF5 datasets for training.
It gets the data matrix from a supported CSV format from either a VFS or downloaded through the network.
```
from cover_class.static.retrieval import generate_hdf5_from_config
generate_hdf5_from_config('/path/to/my/config.yml')
```

### Reporting
Post-training, it is useful to standardize the training and evaluation metadata and metrics. As such, this `cover-class` repository offers a function to automatically generate a PDF report to compare against other models.
Takes in:
- Training:
    - a dictionary to turn into a JSON
    - a dictionary to turn into plots
    - a list of self-made matplotlib figures to include
- Evaluation:
    - a dictionary to turn into a JSON
    - a dictionary to turn into plots
    - a list of self-made matplotlib figures to include
```
```