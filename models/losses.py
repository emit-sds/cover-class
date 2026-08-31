"""Loss functions for fractional-cover model training.

Two self-contained losses, ported from the reference SpecTf training scripts:

- ``FocalLoss`` — multi-label presence (classification). Focal-weighted binary
  cross-entropy on per-class logits; down-weights easy examples so the model
  focuses on the hard, rare-present classes.
- ``FocalCategoricalCrossEntropy`` — fractional unmixing (regression). Focal
  categorical cross-entropy against a fractional (sums-to-1) target over the
  class simplex.

Both accept raw logits (no activation applied). Passing ``alpha=None`` with
``gamma=0`` reduces each to its plain (BCE / categorical CE) form, and the
training script uses the built-in ``BCEWithLogitsLoss`` / ``KLDivLoss`` in that
case instead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal binary cross-entropy for multi-label presence classification.

    inputs: (B, C) logits (pre-sigmoid). targets: (B, C) in {0, 1}.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class FocalCategoricalCrossEntropy(nn.Module):
    """Focal categorical cross-entropy for fractional unmixing (regression).

    inputs: (B, C) logits (pre-softmax). targets: (B, C) fractional ground
    truth, summing to 1 across classes.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # log-softmax -> categorical cross-entropy against the fractional target
        log_probs = F.log_softmax(inputs, dim=-1)
        ce_loss = -(targets * log_probs).sum(dim=-1)

        # p_t = probability mass the model places on the true distribution
        probs = torch.exp(log_probs)
        p_t = (targets * probs).sum(dim=-1)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.alpha is not None:
            # weight by the dominant class' confidence in the target
            tmax = targets.max(dim=-1)[0]
            alpha_weight = self.alpha * tmax + (1 - self.alpha) * (1 - tmax)
            focal_loss = alpha_weight * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
