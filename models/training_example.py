from typing import Optional
import rich_click as click
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import getpass
import matplotlib.pyplot as plt

from cover_class.train import setup_training_from_config, make_simulation_test_set, make_run_name #type: ignore
from cover_class.static.retrieval import generate_hdf5_from_config #type: ignore
from cover_class.utils import seed as sseed #type: ignore
from cover_class.reporting import ModelConfig, Report #type: ignore

ENV_VAR_PREFIX = 'COVER_CLASS_TRAIN_'

@click.command()
@click.option(
    "--outdir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Output file directory.",
    envvar=f'{ENV_VAR_PREFIX}OUTDIR'
)
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the YAML config for the dataloader.",
    envvar=f'{ENV_VAR_PREFIX}CONFIG'
)
@click.option(
    "--lr",
    required=False,
    type=float,
    default=3e-4,
    help="Learning rate.",
    envvar=f'{ENV_VAR_PREFIX}_LR'
)
@click.option(
    "--batch-size",
    required=True,
    type=int,
    help="Batch size.",
    envvar=f'{ENV_VAR_PREFIX}_BSZ'
)
@click.option(
    "--max-training-steps",
    required=False,
    type=int,
    default=np.iinfo(np.uint64).max,
    help="Maximum number of training steps to do. This is needed for simulated data as otherwise there would be no loop termination condition.",
    envvar=f'{ENV_VAR_PREFIX}_MAX_STEPS'
)
@click.option(
    "--log-interval",
    required=False,
    type=int,
    default=None,
    help="Number of steps between logging",
    envvar=f'{ENV_VAR_PREFIX}_LOG_INTERVAL'
)
@click.option(
    "--static-config",
    required=False,
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to the YAML config for generating an HDF5 from a set of CSVs. If provided, it will run a generator function for creating the HDF5s, but if omitted, it will not.",
    envvar=f'{ENV_VAR_PREFIX}_STATIC_CONFIG'
)
@click.option(
    "--simulated-test-set-size",
    required=False,
    type=int,
    default=1_000_000,
    help="Number of rows to generate for the simulated test set.",
    envvar=f'{ENV_VAR_PREFIX}_SIMULATED_TEST_SET_SIZE'
)
@click.option(
    "--seed",
    required=False,
    type=int,
    default=42,
    help="Seed.",
    envvar=f'{ENV_VAR_PREFIX}_SEED'
)
def run_pipeline_classifier(
        outdir: str,
        config: str,
        lr: float,
        batch_size: int,
        max_training_steps: int = np.iinfo(np.uint64).max,
        log_interval: Optional[int] = None,
        static_config: Optional[str] = None,
        simulated_test_set_size: int = 1_000_000,
        seed: int = 42,
    ):

    if static_config:
        print("Creating HDF5s from config")
        generate_hdf5_from_config(static_config)
    
    run_name = make_run_name()
    dataloader, test_X, test_Y = setup_training_from_config(config, batch_size, True, seed, outdir, run_name)

    ## create simulation eval set
    sseed(seed)
    simulation_x_test, simulation_y_test, _ = make_simulation_test_set(dataloader, test_X, test_Y, simulated_test_set_size)

    class MultiLabelClassifier(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.1),

                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1),

                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),

                nn.Linear(32, num_classes)
            )

        def forward(self, x) -> torch.Tensor:
            return self.fc(x)

    num_classes = len(torch.unique(test_Y))
    model       = MultiLabelClassifier(input_dim=test_X.shape[1], num_classes=num_classes)
    criterion   = nn.BCEWithLogitsLoss()
    optimizer   = optim.AdamW(model.parameters(), lr=lr)

    accumulator = 0
    losses = {"Welford arithmetic mean": 0., "loss": []}

    ## NOTE: It's important to instantiate the report object before the training run
    report = Report(
        outdir=outdir,
        config=config,
        author=getpass.getuser(),
        model_config=ModelConfig(
            model=model,
            model_name=MultiLabelClassifier.__name__,
            hyperparams={
                "learning_rate": lr,
                "batch_size": batch_size,
                "optimizer": optim.AdamW.__name__,
                "dims": [test_X.shape[1], 128, 64, 32, num_classes]
            },
        ),
        Y_test=simulation_y_test,
        random_seed=seed,
        run_name=run_name,
    )
    
    for batch_X, batch_Y in dataloader:
        accumulator += 1
        if accumulator == max_training_steps:
            break

        optimizer.zero_grad(set_to_none=True)
        logits: torch.Tensor = model(batch_X)
        loss: torch.Tensor = criterion(logits, batch_Y)
        loss.backward()
        optimizer.step()
    
        nats = loss.item()

        losses["loss"].append(nats) #type: ignore
        losses["Welford arithmetic mean"] = losses["Welford arithmetic mean"] + (nats - losses["Welford arithmetic mean"])/accumulator # type: ignore
        
        if log_interval is not None and (accumulator % log_interval == 0 or accumulator == 0):
            print(f"Step {accumulator:>7} | BCE loss: {nats:.4f} nats | Running average mean: {losses["Welford arithmetic mean"]:.4f}")
    
    ## Save model
    torch.save(model.state_dict(), os.path.join(outdir, 'model.pth'))
    model.eval()

    ## Generate a training loss plot in the report as well
    fig, ax = plt.subplots()
    ax.plot(losses["loss"]) # type: ignore
    ax.set_title("Training Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("BCE nats")
    report.train_figures.append(fig)

    ## Finally, generate the report
    with torch.no_grad():
        y_hat = torch.sigmoid(model(simulation_x_test))
    thresholds = [0.5] * y_hat.shape[-1]
    report.make_report(y_hat, thresholds)

if __name__ == "__main__": 
    run_pipeline_classifier()
