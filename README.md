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
from cover_class.train import make_simulation_test_set

simulated_test_data, simulated_test_labels, simulated_test_fractions = make_simulation_test_set(
    dataloader,
    real_test_data, 
    real_test_labels,
    simulated_test_set_n_rows = <some_number>,
    seed = 42,
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
- Model:
    - a model configuration with the model itself
    - a dictionary of hyperparameter configs
- Training:
    - a dictionary to turn into a JSON
    - a dictionary to turn into plots
    - a list of self-made matplotlib figures to include
- Evaluation:
    - a dictionary to turn into a JSON
    - a dictionary to turn into plots
    - a list of self-made matplotlib figures to include
- A number of other parameters to properly generate the report

The Report can be generated once a context block is finished, so once your training loop exists, the report will be generated. This way, you don't need to explicitly call a generation function. 

> [!IMPORTANT]
> This report relies on setting up a `.netrc` file in the `cover-class/src/cover_class/reporting/assets/.netrc` file path. There's no real good way to avoid it. When a `Report` object is instanited, the control logic will try to download the qualitative assessment files, so there will be an early error if this is not possible. The file download only occurs whenever there aren't the detected files in `cover-class/src/cover_class/reporting/qualitative`.
>
> Steps:
> 1. Set up an account at [https://urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov)
> 2. Copy your username and password to replace `example@email.com` and `mySuperSecurePassword123` in `cover-class/src/cover_class/reporting/assets/.netrc`.

Example of generating a report:
```
# the dataloader, and train/test sets have been generated already

model = ...
dataloader = ...
test_data, test_labels = ...

from cover_class.reporting import ModelConfig, Report

with Report(
        outdir=".",
        config="/path/to/dataloader.yml",
        author="Shun-Ichi Amari",
        model_config=ModelConfig(
            model=GPT6(n_channels, classes),
            model_name="GPT-6",
            hyperparams={
                "learning_rate": 0.001,
                "batch_size": 32,
                "optimizer": "Muon",
            },
        ),
        classification_threshold=[0.3 for _ in range(num_classes)],
        X_test=test_data,
        Y_test=test_labels,
        random_seed=42,
        notes="Welcome to my report!"
    ) as report:
    
    # now do the training
    for X, Y in dataloader:
        model.train()

        # you can also add in logs to the report
        report.train_metric_table.update(model.generate_some_step_metric())

    # or add in any figures
    report.train_figures.append(
        make_some_really_import_figure_I_want_to_show_from_my_training_logs(model, train_data)
    )

    # and then add in any testing figures or metrics to the report as well
    report.test_figures.append(
        some_cool_test_figure_generator_from_my_data(model, test_data)
    )
    report.test_metric_table.update(
        {'import test metric': generate_test_metric(model, test_data)}
    )

    # then exit the context block....
# and the report will be generated here!
```
