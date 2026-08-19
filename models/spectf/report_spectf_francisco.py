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
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

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


def _plot_spectra_group(ax, wavelengths, group_spectra, color, label, alpha=0.5):
    """Plot a group of spectra on *ax* as semi-transparent *color* lines.

    Uses a LineCollection for efficiency, falling back to individual lines
    when the group is empty.
    """
    if len(group_spectra) == 0:
        return

    n_spectra = len(group_spectra)
    # Broadcast wavelengths to match spectra shape: (n_spectra, n_bands)
    wavelengths_bc = np.broadcast_to(wavelengths, group_spectra.shape)
    # Build line segments: shape (n_spectra, n_bands, 2) with (wavelength, value)
    segs = np.stack([wavelengths_bc, group_spectra], axis=-1)  # (n_spectra, n_bands, 2)

    lc = LineCollection(segs, colors=color, alpha=alpha, linewidths=0.5)
    ax.add_collection(lc)
    ax.set_xlim(450, 2500)
    ax.set_ylim(0, 1)


def _shade_dropped_bands(ax, drop_wl_ranges, xlim=(450, 2500)):
    """Shade *drop_wl_ranges* on *ax* as a light gray background, clipped to *xlim*."""
    if not drop_wl_ranges:
        return
    for low, high in drop_wl_ranges:
        if high < xlim[0] or low > xlim[1]:
            continue
        ax.axvspan(max(low, xlim[0]), min(high, xlim[1]), color='gray', alpha=0.15, zorder=0)


def plot_spectra_by_class(spectra, banddef, y_pred, class_names, thresholds, outdir,
                          figure_prefix='spectra', timestamp=None, drop_wl_ranges=None):
    """Generate one figure per class, plotting spectra colored by prediction vs threshold.

    For classes where *all* spectra contain some fraction (soil, pv, npv):
      - green if y_pred >= threshold  (true positive)
      - red   if y_pred <  threshold  (false negative)

    For other classes (snow+ice, water):
      - red only if y_pred >= threshold (false positive)
    """
    # Classes that are always present in this dataset (green/red logic)
    always_present = {'soil', 'pv', 'npv'}
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # banddef is already in nm (from the CSV header wavelengths)
    wavelength_nm = banddef

    for ci, class_name in enumerate(class_names):
        threshold = thresholds[ci]
        y_true_col = y_pred[:, ci]

        print(class_name, threshold)

        if class_name in always_present:
            # Green: prediction >= threshold (true positive)
            # Red:   prediction <  threshold (false negative)
            green_mask = y_true_col >= threshold
            red_mask = ~green_mask
        else:
            # Only plot red (false positives: prediction >= threshold)
            green_mask = np.zeros(len(y_true_col), dtype=bool)
            if threshold == 0.0:
                red_mask = y_true_col > threshold  # strictly greater than 0
            else:
                red_mask = y_true_col >= threshold
        print(y_true_col[:5])
        print(red_mask.sum())

        fig, ax = plt.subplots(figsize=(12, 6))
        _shade_dropped_bands(ax, drop_wl_ranges)

        if green_mask.any():
            _plot_spectra_group(
                ax, wavelength_nm, spectra[green_mask],
                color='green', label=f'Pred >= {threshold:.3f} (true positive)'
            )
        if red_mask.any():
            _plot_spectra_group(
                ax, wavelength_nm, spectra[red_mask],
                color='red', label=f'Pred {"≤" if class_name in always_present else "≥"} {threshold:.3f}'
            )

        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('Reflectance', fontsize=12)
        ax.set_title(f'{class_name} Spectra', fontsize=14, fontweight='bold')
        ax.set_xlim(450, 2500)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        ax.text(
            0.02, 0.98, f'green: {green_mask.sum()}\nred: {red_mask.sum()}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
        )

        if class_name in always_present:
            legend_elements = [
                Line2D([0], [0], color='green', lw=2, label=f'Pred >= {threshold:.3f} (true positive)'),
                Line2D([0], [0], color='red', lw=2, label=f'Pred < {threshold:.3f} (false negative)'),
            ]
        else:
            legend_elements = [
                Line2D([0], [0], color='red', lw=2, label=f'Pred >= {threshold:.3f} (false positive)'),
            ]

        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        plt.tight_layout()

        output_path = os.path.join(outdir, f"{timestamp}_{figure_prefix}_{class_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Spectra plot saved to: {output_path}")
        plt.close(fig)


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
@click.option(
    "--thresholds",
    required=False,
    type=float,
    nargs=5,
    default=[1.0, 0.959, 0.408, 1.000, 0.000],
    help="List of 5 thresholds, one per class, in order: soil, pv, npv, snow+ice, water.",
    envvar=f'{ENV_VAR_PREFIX}THRESHOLDS'
)
def run_francisco_evaluation(
        outdir: str,
        data_config: str,
        model_config: str,
        model_weights: str,
        rfl_csv: str,
        frac_csv: str,
        thresholds: list,
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
            probs[probs < 1e-6] = 0.0
            probs[probs > 1-(1e-6)] = 1.0
            y_pred[i:i+probs.shape[0]] = probs

    y_true = true_fractions

    # Export y_pred as CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    y_pred_csv_path = os.path.join(outdir, f"{timestamp}_y_pred.csv")
    header = ['plot_num'] + class_names
    with open(y_pred_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, plot_id in enumerate(plot_ids):
            writer.writerow([plot_id] + [f'{v:.8f}' for v in y_pred[idx]])
    print(f"\n{'='*80}")
    print(f"y_pred exported to: {y_pred_csv_path}")
    print(f"{'='*80}\n")

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

    # Save figure (reusing the timestamp from the CSV export for consistency)
    output_path = os.path.join(outdir, f"{timestamp}_francisco_scatter_plots.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Scatter plot saved to: {output_path}")
    print(f"{'='*80}\n")

    plt.show()

    # Plot spectra by class with thresholds
    thresholds_list = list(thresholds)
    print(f"\nClass thresholds: {dict(zip(class_names, thresholds_list))}")
    assert len(thresholds_list) == len(class_names), \
        f"Expected {len(class_names)} thresholds, got {len(thresholds_list)}"

    print("Generating spectra plots by class...")
    plot_spectra_by_class(
        spectra=spectra,
        banddef=banddef,
        y_pred=y_pred,
        class_names=class_names,
        thresholds=thresholds_list,
        outdir=outdir,
        timestamp=timestamp,
        drop_wl_ranges=drop_wl_ranges,
    )
    print("Spectra plots complete.")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    run_francisco_evaluation()
