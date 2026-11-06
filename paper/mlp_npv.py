"""MLP-NPV presence classifier.

Verbatim port of SpectralMLPClassifierNPV / make_mlp from
EMIT-FC/frac-cover/frac_cover/models/mlp.py (Y. Marchetti), so the paper
pipeline does not depend on the frac_cover package or its old cover_class
simulator imports.
"""

import torch.nn as nn


def make_mlp(input_dim, hidden_layers, start_width, width_decay=0.5,
             min_width=32, dropout=0.0):
    layers = []
    in_dim, width = input_dim, start_width
    for _ in range(hidden_layers):
        out_dim = int(width)
        layers.extend([nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim),
                       nn.ReLU(), nn.Dropout(dropout)])
        in_dim = out_dim
        width = max(width * width_decay, min_width)
    return nn.Sequential(*layers), in_dim


class SpectralMLPClassifierNPV(nn.Module):
    """Shared MLP head for all classes; a deeper auxiliary branch supplies NPV."""

    def __init__(self, input_dim, num_classes, npv_idx,
                 hidden_layers=2, start_width=64, width_decay=0.5, min_width=32,
                 dropout=0.0, npv_hidden_layers=4, npv_start_width=128,
                 npv_width_decay=0.5, npv_dropout=0.1, npv_min_width=64):
        super().__init__()
        self.npv_idx = npv_idx
        self.shared_backbone, shared_out = make_mlp(
            input_dim, hidden_layers, start_width, width_decay, min_width, dropout)
        self.shared_head = nn.Linear(shared_out, num_classes)
        self.npv_backbone, npv_out = make_mlp(
            input_dim, npv_hidden_layers, npv_start_width, npv_width_decay,
            npv_min_width, npv_dropout)
        # auxiliary branch predicts all classes
        self.npv_aux_head = nn.Linear(npv_out, num_classes)

    def forward(self, x, return_aux=False):
        shared_logits = self.shared_head(self.shared_backbone(x))
        aux_logits = self.npv_aux_head(self.npv_backbone(x))
        final_logits = shared_logits.clone()
        final_logits[:, self.npv_idx] = aux_logits[:, self.npv_idx]
        if return_aux:
            return final_logits, aux_logits
        return final_logits
