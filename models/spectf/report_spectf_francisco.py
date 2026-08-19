import os

# Set CUBLAS_WORKSPACE_CONFIG for deterministic CUDA behavior
# pylint: disable=wrong-import-position
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import csv
from datetime import datetime
import rich_click as click
import yaml

import numpy as np
import torch
import matplotlib.pyplot as plt

from cover_class.utils import read_config
from cover_class.subsample.forward_pipeline import drop_bad_bands, drop_bad_banddef

from spectf.model import SpecTfEncoder
from spectf.utils import get_device

ENV_VAR_PREFIX = 'COVER_CLASS_FRANCISCO_'

# The francisco fraction CSV only reports these material fractions (soil/pv/npv).
# True water and snow+ice are always 0 for this dataset; shade is ignored.
FRACTION_CSV_COLUMNS = {'soil': 'soil', 'pv': 'pv', 'npv': 'npv'}


def load_francisco_data(rfl_csv: str, frac_csv: str, class_names: list, drop_wl_ranges):
    """Load the francisco spectra and align them to their per-plot true fractions.

    Each plot in the fraction CSV maps to multiple spectra in the reflectance CSV
    (matched by the plot identifier), so those spectra share identical labels.

    Returns:
        spectra: (N, B) reflectance with bad bands dropped
        banddef: (B,) wavelengths with bad bands dropped
        true_fractions: (N, C) true fraction per class (water/snow+ice forced to 0)
        plot_ids: (N,) plot identifier per spectrum
    """
    # --- Reflectance CSV: header row is [plot_num, wl_0, wl_1, ...] ---
    with open(rfl_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        wavelengths = np.array([float(w) for w in header[1:]], dtype=np.float64)

        plot_ids = []
        spectra = []
        for row in reader:
            if not row:
                continue
            plot_ids.append(row[0])
            spectra.append([float(v) for v in row[1:]])

    spectra = np.array(spectra, dtype=np.float32)
    plot_ids = np.array(plot_ids)

    # Drop bad bands to match the band definition the model was trained on
    spectra = drop_bad_bands(spectra, wavelengths, drop_wl_ranges)
    banddef = drop_bad_banddef(wavelengths, drop_wl_ranges)

    # --- Fraction CSV: one row per plot ---
    plot_to_frac = {}
    with open(frac_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frac = np.zeros(len(class_names), dtype=np.float32)
            for ci, cname in enumerate(class_names):
                col = FRACTION_CSV_COLUMNS.get(cname)
                if col is not None and col in row:
                    frac[ci] = float(row[col])
                # else: leave at 0 (water, snow+ice)
            plot_to_frac[row['plot']] = frac

    # Align each spectrum to its plot's fractions
    missing = sorted({p for p in plot_ids if p not in plot_to_frac})
    if missing:
        raise ValueError(f"{len(missing)} plot(s) in reflectance CSV have no fraction row: {missing[:5]}...")

    true_fractions = np.stack([plot_to_frac[p] for p in plot_ids], axis=0)

    return spectra, banddef, true_fractions, plot_ids


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
    help="Path to the YAML data config (used for class names and drop-bands).",
    envvar=f'{ENV_VAR_PREFIX}DATA_CONFIG'
)
@click.option(
    "--model-config",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the YAML config for the model architecture.",
    envvar=f'{ENV_VAR_PREFIX}MODEL_CONFIG'
)
@click.option(
    "--model-weights",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the model weights file (.pth).",
    envvar=f'{ENV_VAR_PREFIX}MODEL_WEIGHTS'
)
@click.option(
    "--rfl-csv",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to emit_pixels_rfl.csv (spectra).",
    envvar=f'{ENV_VAR_PREFIX}RFL_CSV'
)
@click.option(
    "--frac-csv",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to fraction_output.csv (per-plot fractions).",
    envvar=f'{ENV_VAR_PREFIX}FRAC_CSV'
)
def run_francisco_evaluation(
        outdir: str,
        data_config: str,
        model_config: str,
        model_weights: str,
        rfl_csv: str,
        frac_csv: str,
    ):

    # Load model config
    with open(model_config, 'r', encoding='utf-8') as f:
        m_config = yaml.safe_load(f)

    # Load data config (for class names and drop-bands)
    d_config = read_config(data_config)
    ds = d_config['datasets']
    class_names = [str(c) for c in ds.keys() if ds[c] is not None and len(ds[c])]
    print(f"Classes: {class_names}")
    drop_wl_ranges = d_config['drop-bands-wavelengths']

    # Load francisco spectra and align to per-plot fractions
    print(f"Loading spectra from {rfl_csv}")
    print(f"Loading fractions from {frac_csv}")
    spectra, banddef, true_fractions, plot_ids = load_francisco_data(
        rfl_csv, frac_csv, class_names, drop_wl_ranges
    )
    print(f"Loaded {spectra.shape[0]} spectra ({spectra.shape[1]} bands) "
          f"across {len(np.unique(plot_ids))} plots")

    # Device (cuda / mps / cpu)
    device = get_device(0)

    # Model definition
    banddef_t = torch.from_numpy(np.asarray(banddef)).to(dtype=torch.float32, device=device)
    model = SpecTfEncoder(banddef_t,
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

    # Inference — this is a BCE (sigmoid) classifier, so apply sigmoid to logits
    bs = m_config['batch_size']
    n = spectra.shape[0]
    y_pred = np.zeros((n, m_config['model']['dim_output']), dtype=float)

    print("Running inference...")
    spectra_t = torch.from_numpy(spectra).to(dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, n, bs):
            batch = spectra_t[i:i+bs].to(device=device)
            batch = torch.unsqueeze(batch, -1)
            logits = model(batch)
            probs = torch.sigmoid(logits).detach().cpu().numpy().astype(float)
            y_pred[i:i+probs.shape[0]] = probs

    y_true = true_fractions

    # Scatter plot: one panel per class, true fraction (x) vs predicted posterior (y)
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))
    if n_classes == 1:
        axes = [axes]

    for i, (ax, class_name) in enumerate(zip(axes, class_names)):
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.3, s=8, c='blue', edgecolors='none')

        # 1:1 line
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='1:1 line')

        ax.set_xlabel('True Fraction', fontsize=12)
        ax.set_ylabel('Predicted Posterior', fontsize=12)
        ax.set_title(f'{class_name}', fontsize=14, fontweight='bold')

        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_aspect('equal')

        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(outdir, f"{timestamp}_francisco_scatter_plots.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Scatter plot saved to: {output_path}")
    print(f"{'='*80}\n")

    plt.show()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    run_francisco_evaluation()
