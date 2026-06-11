import os

# Set CUBLAS_WORKSPACE_CONFIG for deterministic CUDA behavior
# pylint: disable=wrong-import-position
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import getpass
from datetime import datetime
import rich_click as click
import yaml

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from cover_class.train import setup_training_from_config, make_simulation_test_set, banddef_from_config #type: ignore
from cover_class.utils import seed as sseed, ood_test_set_from_config #type: ignore
from cover_class.reporting import ModelConfig, Report #type: ignore
from cover_class.simulation.force_fractions import ForcedFractionSimulation

from spectf.model import SpecTfEncoder
from spectf.utils import get_device

ENV_VAR_PREFIX = 'COVER_CLASS_TRAIN_'

class TestDataset(Dataset):
    def __init__(self, test_X, test_Y):
        super().__init__()

        self.test_X = test_X
        self.test_Y = test_Y
    
    def __len__(self):
        return len(self.test_Y)

    def __getitem__(self, idx):
        return self.test_X[idx], self.test_Y[idx]

@click.command()
@click.option(
    "--outdir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Output file directory.",
    envvar=f'{ENV_VAR_PREFIX}OUTDIR'
)
@click.option(
    "--data-config",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the YAML config for the dataloader.",
    envvar=f'{ENV_VAR_PREFIX}_DATA_CONFIG'
)
@click.option(
    "--model-config",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the YAML config for the model architecture.",
    envvar=f'{ENV_VAR_PREFIX}_MODEL_CONFIG'
)
@click.option(
    "--model-weights",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the model weights file (.pth).",
    envvar=f'{ENV_VAR_PREFIX}_MODEL_WEIGHTS'
)
@click.option(
    "--simulated-test-set-size",
    required=False,
    type=int,
    default=100_000,
    help="Number of rows to generate for the simulated test set.",
    envvar=f'{ENV_VAR_PREFIX}_SIMULATED_TEST_SET_SIZE'
)
@click.option(
    "--export",
    is_flag=True,
    default=False,
    help="Export y_hat_ood to an HDF5 file in the output directory.",
)
@click.option(
    "--ood-overfit",
    is_flag=True,
    default=False,
    help="Calculate the OOD metrics using the best threshold, not the simulated test threshold"
)
@click.option(
    "--fixed-fpr",
    type=float,
    default=None,
    help="Calculate the metrics at the fixed FPR, not for the best F1 score"
)
def run_report_generator(
        outdir: str,
        data_config: str,
        model_config: str,
        model_weights: str,
        simulated_test_set_size: int = 100_000,
        export: bool = False,
        ood_overfit: bool = False,
        fixed_fpr: float = None,
    ):

    # Load model config
    with open(model_config, 'r', encoding='utf-8') as f:
        m_config = yaml.safe_load(f)

    # Set up dataloader and test sets
    dataloader, test_X, test_Y = setup_training_from_config(
        data_config,
        m_config['batch_size'],
        shuffle=True,
        seed=m_config['random_seed'],
        subsampled_files_outdir=outdir,
        misc_dataloader_params={'num_workers': m_config['training']['num_workers']})

    # Get banddef (wavelength definitions) from the hdf5 files specified in the data config
    banddef = banddef_from_config(data_config)

    # Create the simulated test set
    sseed(m_config['random_seed'])
    simulation_x_test, simulation_y_test, _ = make_simulation_test_set(dataloader, test_X, test_Y, simulated_test_set_size)

    # Create a dataset/dataloader to feed the test set in batches
    test_dataset = TestDataset(simulation_x_test, simulation_y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=m_config['batch_size'], shuffle=False)

    # Create a dataset/dataloader to feed the OOD validation set in batches
    ood_test_set_x, ood_test_set_y = ood_test_set_from_config(data_config, include_unknown=False)
    ood_dataset = TestDataset(ood_test_set_x, ood_test_set_y)
    ood_dataloader = DataLoader(ood_dataset, batch_size=m_config['batch_size'], shuffle=False)

    # hardcoded GPU 0
    device = get_device(0)

    # model definition
    model = SpecTfEncoder(banddef.to(dtype=torch.float32, device=device),
                         dim_output=m_config['model']['dim_output'],
                         num_heads=m_config['model']['num_heads'],
                         dim_proj=m_config['model']['dim_proj'],
                         dim_ff=m_config['model']['dim_ff'],
                         dropout=m_config['model']['dropout'],
                         agg=m_config['model']['agg'],
                         use_residual=m_config['model']['use_residual'],
                         num_layers=m_config['model']['num_layers']).to(device)

    # Load model weights
    print(f"Loading model weights from {model_weights}")
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.eval()

    # Generate timestamp for report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = Report(
        outdir=outdir,
        config=data_config,
        author=getpass.getuser(),
        model_config=ModelConfig(
            model=model,
            model_name=SpecTfEncoder.__name__,
            hyperparams={
                "learning_rate": m_config['training']['learning_rate'],
                "batch_size": m_config['batch_size'],
                "optimizer": "AdamWScheduleFree",
                "params": m_config['model']
            },
        ),
        Y_test=simulation_y_test,
        Y_ood_test=ood_test_set_y,
        random_seed=m_config['random_seed'],
        run_name=timestamp
    )

    # batch size
    bs = m_config['batch_size']

    # Test loop
    print("Evaluating on simulated test set...")
    y_hat = np.zeros_like(simulation_y_test, dtype=float)
    for i, (batch_X, _) in enumerate(test_dataloader):
        batch_X = batch_X.to(device=device, dtype=torch.float32)
        batch_X = torch.unsqueeze(batch_X, -1)
        with torch.no_grad():
            logits = model(batch_X)
            batch_y_hat = torch.sigmoid(logits)
            batch_y_hat = batch_y_hat.detach().cpu().numpy().astype(float)
            batch_len = len(batch_y_hat)
            y_hat[i*bs:i*bs+batch_len] = batch_y_hat

    # OOD loop
    print("Evaluating on OOD validation set...")
    y_hat_ood = np.zeros_like(ood_test_set_y, dtype=float)
    for i, (batch_X, _) in enumerate(ood_dataloader):
        batch_X = batch_X.to(device=device, dtype=torch.float32)
        batch_X = torch.unsqueeze(batch_X, -1)
        with torch.no_grad():
            logits = model(batch_X)
            batch_y_hat = torch.sigmoid(logits)
            batch_y_hat = batch_y_hat.detach().cpu().numpy().astype(float)
            batch_len = len(batch_y_hat)
            y_hat_ood[i*bs:i*bs+batch_len] = batch_y_hat

    if export:
        export_path = os.path.join(outdir, f"{timestamp}_y_hat_ood.h5")
        print(f"Exporting y_hat_ood to {export_path}...")
        with h5py.File(export_path, "w") as f:
            f.create_dataset("y_hat_ood", data=y_hat_ood.astype(np.float32))

    # Fraction simulation
    ff_simulated_test_set_size = 1000
    fs = ForcedFractionSimulation(dataloader, test_X, test_Y, ff_simulated_test_set_size)
    fs.hard_max_iters = 1_000
    for frac_sim_data, _ in fs:
        if len(frac_sim_data) == 0:
            continue

        frac_sim_y_hat_list = []
        for i in range(0, len(frac_sim_data), bs):
            batch_data = frac_sim_data[i : i + bs]
            with torch.no_grad():
                batch_data = batch_data.unsqueeze(-1).to(device=device, dtype=torch.float32)
                batch_y_hat = torch.sigmoid(model(batch_data))
                frac_sim_y_hat_list.append(batch_y_hat.cpu())

        frac_sim_y_hat = torch.cat(frac_sim_y_hat_list, dim=0)
        report.append_fractional_simulation_result(fs, frac_sim_y_hat)

    # Generate report
    report.make_report(
        y_hat,
        y_hat_ood_test=y_hat_ood,
        class_thresholds=None,
        ood_overfit=ood_overfit,
        target_fpr=fixed_fpr
    )

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    run_report_generator()
