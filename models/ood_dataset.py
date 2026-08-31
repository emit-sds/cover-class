"""ood_dataset — load the out-of-distribution (OOD) validation set.

Standalone reimplementation of the old `ood_test_set_from_config` with no
dependency on the `cover_class` package. Reads a labeled OOD h5 file (spectra +
per-class presence/absence/ambiguous labels) and remaps its label columns onto
a caller-supplied class axis.

Class-axis remap: the OOD h5 file stores labels in whatever column order its
`classes` attr says (e.g. `['soil','pv','npv','snow+ice','water']`), which is
NOT the same order as the simulator's canonical axis (`specmix.sim_config`'s
`classes: [water, pv, npv, soil, snow+ice]`). `load_ood` looks up each
caller-requested class name in the file's `classes` attr and reindexes columns
by name, so the returned Y always matches the caller's `classes` argument
regardless of how the file happens to store them.

Label values in the file are {0, 1, 2} = absent / present / ambiguous.
`mask_ambiguous=True` (default) turns 2 into NaN so ambiguous labels can be
excluded from per-class metrics later; `mask_ambiguous=False` treats 2 as
present (1).
"""

import h5py
import numpy as np


def load_ood(path, classes, mask_ambiguous=True):
    """Load the OOD validation set, remapped to the caller's class axis.

    Args:
        path: path to the OOD h5 file.
        classes: list of class names in the caller's canonical order (the
            simulator's axis, e.g. ['water','pv','npv','soil','snow+ice']).
            Output label columns are ordered to match this list.
        mask_ambiguous: if True (default), label value 2 (ambiguous) becomes
            NaN so it can be masked out of per-class metrics later. If False,
            2 is treated as present (1).

    Returns:
        (X, Y):
          X: np.ndarray float32, shape (N, 207) -- spectra as-is.
          Y: np.ndarray float32, shape (N, len(classes)) -- labels remapped to
             the `classes` order; 0/1, with NaN where ambiguous (if masked).
    """
    with h5py.File(path, "r") as f:
        X = f["spectra"][:].astype(np.float32)
        raw_labels = f["labels"][:]
        file_classes = [c.decode() if isinstance(c, bytes) else str(c)
                         for c in f.attrs["classes"]]

    name_to_col = {name: i for i, name in enumerate(file_classes)}

    cols = []
    for name in classes:
        if name not in name_to_col:
            raise ValueError(
                f"Requested class {name!r} not found in OOD file's classes "
                f"{file_classes!r} (path={path})")
        cols.append(name_to_col[name])

    Y = raw_labels[:, cols].astype(np.float32)
    if mask_ambiguous:
        Y = np.where(Y == 2, np.nan, Y).astype(np.float32)
    else:
        Y = np.where(Y == 2, 1.0, Y).astype(np.float32)

    return X, Y


class OODDataset:
    """Map-style dataset over cached OOD (X, Y) numpy arrays.

    Duck-typed like MixtureDataset in this repo (NOT subclassing
    torch.utils.data.Dataset) so torch stays an optional import at module load.
    Returns (spectra_row, label_row) numpy arrays per item; pair it with the
    repo's `numpy_collate` (from specmix) in a DataLoader.
    """

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


if __name__ == "__main__":
    import os

    PATH = "datasets/ood/validation_20260317_wfids_noash_goodwl_subsampled.h5"
    CLASSES = ["water", "pv", "npv", "soil", "snow+ice"]

    if not os.path.exists(PATH):
        raise SystemExit(f"OOD file not found at {PATH!r}; run from repo root.")

    X, Y = load_ood(PATH, CLASSES)
    print(f"X shape: {X.shape}, dtype={X.dtype}")
    print(f"Y shape: {Y.shape}, dtype={Y.dtype}")

    # Cross-check against raw file columns (via the file's own classes attr).
    with h5py.File(PATH, "r") as f:
        raw_labels = f["labels"][:]
        file_classes = [c.decode() if isinstance(c, bytes) else str(c)
                         for c in f.attrs["classes"]]
    name_to_col = {name: i for i, name in enumerate(file_classes)}

    for j, name in enumerate(CLASSES):
        col = Y[:, j]
        n_present = int(np.sum(col == 1))
        n_absent = int(np.sum(col == 0))
        n_nan = int(np.sum(np.isnan(col)))
        raw_col = raw_labels[:, name_to_col[name]]
        n_raw_ambiguous = int(np.sum(raw_col == 2))
        assert n_nan == n_raw_ambiguous, (
            f"{name}: nan count {n_nan} != raw ambiguous count {n_raw_ambiguous}")
        print(f"{name:>10s}: present={n_present:6d} absent={n_absent:6d} "
              f"nan={n_nan:6d} (matches raw ambiguous count {n_raw_ambiguous})")

    ds = OODDataset(X, Y)
    assert len(ds) == X.shape[0]
    x0, y0 = ds[0]
    assert x0.shape == (207,) and y0.shape == (len(CLASSES),)
    print("OODDataset smoke test OK")
