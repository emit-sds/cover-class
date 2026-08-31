"""Loader for the class-presence mixture prior (`mixture_priors.csv`).

This file is the single source of domain truth for WHICH class combinations may
appear in a simulated pixel and HOW OFTEN, relative to each other. The spectral
scientists own the CSV; this loader turns it into a validated presence-pattern
distribution the simulator can sample from.

CSV schema (one row per presence pattern; all 2^n patterns must be enumerated):
  - one 0/1 column per class (multi-hot: 1 = that class is present in the pixel),
  - `weight`  : a non-negative INTEGER relative frequency. 0 = forbidden (the
                combination does not occur); >0 = allowed, larger = more frequent.
                Weights are un-normalized -- the probability is derived here and
                never stored, so there is no second column to keep in sync.
  - `justification` : free-text provenance for the row. Documentation only; not parsed.

Every failure is an explicit error at load time -- nothing is silently clamped or
renormalized. numpy + pandas only; returns plain numpy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

WEIGHT_COL = "weight"
JUSTIFICATION_COL = "justification"
_NON_CLASS_COLS = {WEIGHT_COL, JUSTIFICATION_COL}


@dataclass(frozen=True)
class MixturePrior:
    """A validated class-presence distribution.

    Attributes:
        classes:      class names, in column order (the canonical class axis).
        patterns:     (P, n_classes) int8 multi-hot rows, ALLOWED patterns only
                      (weight > 0). Row p, column c == 1 iff class c is present.
        weights:      (P,) int64 un-normalized weights, aligned with `patterns`.
        probs:        (P,) float64 normalized probabilities (weights / sum),
                      aligned with `patterns`. Derived, never read from disk.
    """
    classes: List[str]
    patterns: np.ndarray
    weights: np.ndarray
    probs: np.ndarray

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def pattern_class_names(self) -> List[List[str]]:
        """Each allowed pattern as the list of present class names (for logging)."""
        return [[c for c, on in zip(self.classes, row) if on]
                for row in self.patterns]


def load_mixture_prior(
    path: str,
    classes: Optional[Sequence[str]] = None,
) -> MixturePrior:
    """Load, validate, and normalize the mixture prior CSV.

    Args:
        path: path to the mixture-priors CSV.
        classes: optional subset/reordering of class names to keep. When given,
            rows that mark ANY dropped class present are removed and the
            remaining weights are renormalized over the survivors. This mirrors
            the "disable a class" behavior of the old pipeline but is EXPLICIT
            and validated (unknown names error; see below), rather than an
            implicit side effect of a config toggle. When None, all CSV classes
            are kept in column order.

    Returns:
        MixturePrior with only allowed (weight > 0) patterns.

    Raises:
        ValueError / FileNotFoundError on any malformed input. This function
        never silently repairs bad data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"mixture prior CSV not found: {path}")

    df = pd.read_csv(path)

    # --- structural validation -------------------------------------------------
    if WEIGHT_COL not in df.columns:
        raise ValueError(f"CSV must have a '{WEIGHT_COL}' column; got {list(df.columns)}")
    if JUSTIFICATION_COL not in df.columns:
        raise ValueError(f"CSV must have a '{JUSTIFICATION_COL}' column; got {list(df.columns)}")

    csv_classes = [c for c in df.columns if c not in _NON_CLASS_COLS]
    if not csv_classes:
        raise ValueError("CSV has no class columns (only weight/justification).")

    n = len(csv_classes)
    expected_rows = 2 ** n
    if len(df) != expected_rows:
        raise ValueError(
            f"expected all {expected_rows} presence patterns for {n} classes "
            f"({csv_classes}), got {len(df)} rows. The prior must enumerate every "
            f"2^n combination so nothing is silently undefined.")

    # class columns must be strictly 0/1
    cls_block = df[csv_classes]
    if not np.isin(cls_block.to_numpy(), (0, 1)).all():
        bad = cls_block.columns[~np.isin(cls_block.to_numpy(), (0, 1)).all(axis=0)].tolist()
        raise ValueError(f"class columns must contain only 0/1; offending columns: {bad}")

    # patterns must be unique (no duplicate/contradictory rows)
    patterns_all = cls_block.to_numpy(dtype=np.int8)
    uniq = np.unique(patterns_all, axis=0)
    if uniq.shape[0] != patterns_all.shape[0]:
        raise ValueError("duplicate presence patterns found; each of the 2^n "
                         "combinations must appear exactly once.")

    # weights must be non-negative integers
    w_raw = df[WEIGHT_COL].to_numpy()
    if not np.all(np.equal(np.mod(w_raw, 1), 0)):
        raise ValueError(f"'{WEIGHT_COL}' must be integers; got non-integer values.")
    weights_all = w_raw.astype(np.int64)
    if (weights_all < 0).any():
        raise ValueError(f"'{WEIGHT_COL}' must be >= 0 (0 = forbidden); found negatives.")

    # --- optional subset selection --------------------------------------------
    if classes is not None:
        classes = list(classes)
        unknown = set(classes) - set(csv_classes)
        if unknown:
            raise ValueError(f"requested classes not in CSV: {sorted(unknown)}; "
                             f"available: {csv_classes}")
        dropped = [c for c in csv_classes if c not in classes]
        keep_col = np.array([c in classes for c in csv_classes])
        # Drop any pattern that marks a dropped class present -- keeping it would
        # mean simulating a mixture that includes a class we're not modeling.
        if dropped:
            drop_idx = np.array([csv_classes.index(c) for c in dropped])
            has_dropped = patterns_all[:, drop_idx].any(axis=1)
        else:
            has_dropped = np.zeros(len(patterns_all), dtype=bool)
        patterns_all = patterns_all[~has_dropped][:, keep_col]
        weights_all = weights_all[~has_dropped]
        # Reorder columns to the requested class order.
        order = [csv_classes.index(c) for c in classes if c in csv_classes]
        # `keep_col` already restricted to kept classes in csv order; remap.
        kept_csv = [c for c in csv_classes if c in classes]
        reorder = [kept_csv.index(c) for c in classes]
        patterns_all = patterns_all[:, reorder]
        out_classes = classes
    else:
        out_classes = csv_classes

    # --- keep allowed patterns, derive probabilities --------------------------
    allowed = weights_all > 0
    if not allowed.any():
        raise ValueError("no allowed patterns (all weights are 0) after loading"
                         + ("/subsetting." if classes is not None else "."))

    patterns = patterns_all[allowed]
    weights = weights_all[allowed]

    # Drop the all-absent pattern if it somehow carries weight -- an empty pixel
    # is not a valid mixture target. (weight 0 already excludes it normally.)
    nonempty = patterns.any(axis=1)
    if not nonempty.all():
        raise ValueError("the all-absent pattern (no classes present) has a "
                         "positive weight; an empty pixel is not a valid mixture.")

    probs = weights / weights.sum()

    return MixturePrior(
        classes=list(out_classes),
        patterns=patterns,
        weights=weights,
        probs=probs.astype(np.float64),
    )


if __name__ == "__main__":
    # Smoke test / human-readable dump of the shipped prior.
    here = os.path.dirname(os.path.abspath(__file__))
    prior = load_mixture_prior(os.path.join(here, "data", "mixture_priors.csv"))
    print(f"classes: {prior.classes}")
    print(f"allowed patterns: {len(prior.patterns)} / {2 ** prior.n_classes}")
    names = prior.pattern_class_names()
    for nm, w, p in sorted(zip(names, prior.weights, prior.probs),
                           key=lambda t: -t[2]):
        print(f"  {p:6.4f}  (w={w:>2d})  {' + '.join(nm)}")
    assert abs(prior.probs.sum() - 1.0) < 1e-9
    print("probs sum to 1.0 ✓")
