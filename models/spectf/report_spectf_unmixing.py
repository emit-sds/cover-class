import os

# Set CUBLAS_WORKSPACE_CONFIG for deterministic CUDA behavior
# pylint: disable=wrong-import-position
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from datetime import datetime
import rich_click as click
import yaml

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from cover_class.train import setup_training_from_config, make_simulation_test_set, banddef_from_config
from cover_class.utils import seed as sseed

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
def run_unmixing_evaluation(
        outdir: str,
        data_config: str,
        model_config: str,
        model_weights: str,
        simulated_test_set_size: int = 100_000,
    ):

    # Load model config
    with open(model_config, 'r', encoding='utf-8') as f:
        m_config = yaml.safe_load(f)

    # Load data config
    with open(data_config, 'r', encoding='utf-8') as f:
        d_config = yaml.safe_load(f)

    # Get class names from data config
    ds = d_config['datasets']
    class_names = [str(c) for c in ds.keys() if ds[c] is not None and len(ds[c])]
    print(f"Classes: {class_names}")

    # Set up dataloader and test sets
    dataloader, test_X, test_Y = setup_training_from_config(
        data_config,
        m_config['batch_size'],
        shuffle=True,
        seed=m_config['random_seed'],
        subsampled_files_outdir=outdir,
        return_fractions=True,  # Enable fraction mode for unmixing
        misc_dataloader_params={'num_workers': m_config['training']['num_workers']})

    # Get banddef (wavelength definitions)
    banddef = banddef_from_config(data_config)

    # Create the simulated test set
    sseed(m_config['random_seed'])
    simulation_x_test, simulation_y_labels, simulation_y_fractions = make_simulation_test_set(
        dataloader, test_X, test_Y, simulated_test_set_size, one_hot_encode=False
    )

    # Create a dataset/dataloader to feed the test set in batches
    test_dataset = TestDataset(simulation_x_test, simulation_y_fractions)
    test_dataloader = DataLoader(test_dataset, batch_size=m_config['batch_size'], shuffle=False)

    # hardcoded GPU 0
    device = get_device(0)

    # Model definition
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

    # Batch size
    bs = m_config['batch_size']

    # Test loop - get predictions
    print("Evaluating on simulated test set...")
    y_hat_fractions = np.zeros_like(simulation_y_fractions, dtype=float)

    for i, (batch_X, _) in enumerate(test_dataloader):
        batch_X = batch_X.to(device=device, dtype=torch.float32)
        batch_X = torch.unsqueeze(batch_X, -1)
        with torch.no_grad():
            logits = model(batch_X)
            # Apply softmax to get fractions
            batch_y_hat = torch.softmax(logits, dim=-1)
            batch_y_hat = batch_y_hat.detach().cpu().numpy().astype(float)
            batch_len = len(batch_y_hat)
            y_hat_fractions[i*bs:i*bs+batch_len] = batch_y_hat

    # Convert to numpy for metrics computation
    y_true = simulation_y_fractions.cpu().numpy() if torch.is_tensor(simulation_y_fractions) else simulation_y_fractions
    y_pred = y_hat_fractions

    # Compute metrics per class
    print("\n" + "="*80)
    print("REGRESSION METRICS PER CLASS")
    print("="*80)

    metrics = {}
    for i, class_name in enumerate(class_names):
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))

        metrics[class_name] = {
            'R²': r2,
            'MAE': mae,
            'RMSE': rmse
        }

        print(f"\n{class_name.upper()}")
        print(f"  R² Score:  {r2:.4f}")
        print(f"  MAE:       {mae:.4f}")
        print(f"  RMSE:      {rmse:.4f}")

    # Create 5-panel scatter plot
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for i, (ax, class_name) in enumerate(zip(axes, class_names)):
        # Scatter plot
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.3, s=1, c='blue', edgecolors='none')

        # 1:1 line
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='1:1 line')

        # Labels and title
        ax.set_xlabel('True Fraction', fontsize=12)
        ax.set_ylabel('Predicted Fraction', fontsize=12)
        ax.set_title(f'{class_name}', fontsize=14, fontweight='bold')

        # Set axis limits
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')

        # Add metrics as text
        metrics_text = (
            f"R² = {metrics[class_name]['R²']:.3f}\n"
            f"MAE = {metrics[class_name]['MAE']:.3f}\n"
            f"RMSE = {metrics[class_name]['RMSE']:.3f}"
        )
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Grid
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(outdir, f"{timestamp}_unmixing_scatter_plots.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Scatter plot saved to: {output_path}")
    print(f"{'='*80}\n")

    # Save metrics to text file
    metrics_path = os.path.join(outdir, f"{timestamp}_unmixing_metrics.txt")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("REGRESSION METRICS PER CLASS\n")
        f.write("="*80 + "\n\n")
        for class_name in class_names:
            f.write(f"{class_name.upper()}\n")
            f.write(f"  R² Score:  {metrics[class_name]['R²']:.6f}\n")
            f.write(f"  MAE:       {metrics[class_name]['MAE']:.6f}\n")
            f.write(f"  RMSE:      {metrics[class_name]['RMSE']:.6f}\n\n")

    print(f"Metrics saved to: {metrics_path}")

    plt.show()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    run_unmixing_evaluation()
