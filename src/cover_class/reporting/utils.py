from typing import List, Optional, Any
import torch
import numpy as np
import matplotlib.pyplot as plt 
import math

from cover_class.utils import load_rfl
from cover_class.subsample.forward_pipeline import drop_bad_bands

def inference_over_scene(
        rgb_fp:str,
        hdr_fp:str,
        model:Any, 
        drop_wl_ranges:Optional[List[List[int]]]=None
    ):
    rfl, bands = load_rfl(hdr_fp)
    original_shape = rfl.shape
    rfl = rfl.reshape((original_shape[0] * original_shape[1], original_shape[2]))
    rfl = drop_bad_bands(rfl, bands, drop_wl_ranges)
    posterior: np.ndarray
    if isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            posterior_t: torch.Tensor = model(torch.from_numpy(rfl).to(dtype=torch.float32))
        posterior = posterior_t.cpu().detach().numpy()
    else:
        posterior = model(rfl)
    posterior = posterior.reshape((original_shape[0], original_shape[1], posterior.shape[-1]))

    num_classes: int = list(posterior.shape)[-1]
    
    for i in range(num_classes):
        #TODO: Make a matplotlib heatmap image of the scene with the probabilities.
        #   each plot needs to be color coded for their own class
        #   The plot needs to be 2 columns by N rows. RGB image needs to be in the top left of the image.
        #   There should also be a heatmap colorbar on the right of each subfigure heatmap. BUT MAKE SURE THEY'RE ALL NORMALIZED FROM 0 to 1
        c = posterior[:, :, i]


    

    # ----------------------------------------------------------------------
    # 2. Build subplot grid (2 columns, ceil((num_classes + 1) / 2) rows)
    #    Total panels = 1 RGB + num_classes heatmaps
    # ----------------------------------------------------------------------
    total_panels = 1 + num_classes
    ncols = 2
    nrows = math.ceil(total_panels / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 4 * nrows))
    # `axes` might be 2D or 1D depending on nrows; normalize to flat list
    if isinstance(axes, np.ndarray):
        axes_flat = axes.ravel()
    else:
        axes_flat = [axes]

    # ----------------------------------------------------------------------
    # 3. Top-left: RGB image
    # ----------------------------------------------------------------------
    rgb_img = plt.imread(rgb_fp)
    ax_rgb = axes_flat[0]
    ax_rgb.imshow(rgb_img)
    ax_rgb.set_title("RGB Image")
    ax_rgb.axis("off")

    # ----------------------------------------------------------------------
    # 4. Heatmaps for each class
    # ----------------------------------------------------------------------
    for class_idx in range(num_classes):
        ax_idx = class_idx + 1  # +1 because 0 is RGB
        if ax_idx >= len(axes_flat):
            break  # safety

        ax = axes_flat[ax_idx]
        prob_map = posterior[:, :, class_idx]

        # Heatmap normalized between 0 and 1
        im = ax.imshow(prob_map, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(f"Class {class_idx} Probabilities")
        ax.axis("off")

        # Individual colorbar on the right of each heatmap
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Probability", rotation=90)

    # ----------------------------------------------------------------------
    # 5. Hide any unused subplots
    # ----------------------------------------------------------------------
    for idx in range(total_panels, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.tight_layout()
    return fig
