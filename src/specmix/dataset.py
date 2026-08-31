"""MixtureDataset — map-style PyTorch Dataset wrapping the MixtureSimulator.

Each __getitem__ is one independent mixture, so multi-worker DataLoaders
parallelize for free. Per-item RNG is derived from (base_seed, global_index),
where global_index = epoch * epoch_size + index is monotonic across the whole
run. So every item is reproducible and stable: index i maps to the same mixture
regardless of num_workers (the DataLoader's sampler, not the seed, guarantees
each index is fetched once per epoch, so there are never duplicate spectra).

The simulator returns (spectra, fractions). Fractions ARE the regression target;
for classification the caller opts into `fractions_to_presence` (a threshold).

Torch is imported lazily so `simulator.py` and the selftest stay Torch-free.
"""

import numpy as np

from .simulator import MixtureSimulator, DEFAULT_CONFIG


def numpy_collate(batch):
    """Stack (spectra, fractions) tuples into two numpy->torch batch tensors.

    Module-level (not a closure) so it pickles for spawn-based DataLoader workers
    (macOS default start method).
    """
    import torch
    X = np.stack([b[0] for b in batch])
    F = np.stack([b[1] for b in batch])
    return torch.from_numpy(X), torch.from_numpy(F)


def fractions_to_presence(fractions, threshold=0.0):
    """Binarize a fraction vector/array to 0/1 presence at `threshold`.

    Caller-side helper for classification targets. threshold=0.0 means "any
    nonzero fraction is present"; set higher to only count classes above a
    minimum cover.
    """
    return (np.asarray(fractions) > threshold).astype(np.float32)


class MixtureDataset:
    """Map-style dataset of simulated mixtures.

    Duck-typed (not subclassed from torch.utils.data.Dataset) so the module stays
    Torch-optional; DataLoader accepts it. Mixtures are generated on the fly, so
    `epoch_size` just sets how many items the DataLoader yields per epoch.
    """

    def __init__(self, config=None, epoch_size=None, base_seed=0,
                 split=None, split_frac=0.5, split_seed=0):
        self.sim = MixtureSimulator(config, split=split, split_frac=split_frac,
                                    split_seed=split_seed)
        self.base_seed = int(base_seed)
        if epoch_size is None:
            epoch_size = self.sim.cfg.get("dataset", {}).get("epoch_size", 100000)
        self.epoch_size = int(epoch_size)
        # Slides the global-index window forward each epoch (see set_epoch, _rng).
        self.epoch = 0

    def set_epoch(self, epoch):
        """Advance the global-index window so the next epoch draws fresh mixtures.

        Call BEFORE building the epoch's DataLoader iterator, so freshly spawned
        workers observe it. Incompatible with persistent_workers=True (persisted
        workers keep their original epoch and would repeat the same draw).
        """
        self.epoch = int(epoch)

    @property
    def wavelengths(self):
        """207 good-band wavelengths (the model's band definition)."""
        return self.sim.wavelengths

    def __len__(self):
        return self.epoch_size

    def _rng(self, index):
        # Seed from (base_seed, global_index): one run-long stream, so index i
        # maps to the same mixture regardless of num_workers. No worker_id — the
        # sampler already fetches each index once per epoch, so no duplicates.
        global_index = self.epoch * self.epoch_size + index
        ss = np.random.SeedSequence([self.base_seed, global_index])
        return np.random.default_rng(ss)

    def __getitem__(self, index):
        spectra, fractions = self.sim.simulate(self._rng(index))
        return spectra, fractions


if __name__ == "__main__":
    # Multi-worker smoke test: distinct mixtures per item, no cross-worker dupes.
    import torch
    from torch.utils.data import DataLoader

    ds = MixtureDataset(DEFAULT_CONFIG, epoch_size=64)

    # Fetch items in index order (shuffle=False) so we can compare index->spectrum
    # across worker counts, not just uniqueness within one setting.
    by_workers = {}
    for nw in (0, 4):
        dl = DataLoader(ds, batch_size=16, num_workers=nw, shuffle=False,
                        collate_fn=numpy_collate)
        seen = []
        for X, F in dl:
            assert X.shape[1] == ds.sim.n_bands
            assert F.shape[1] == ds.sim.n_classes
            assert torch.allclose(F.sum(1), torch.ones(len(F)), atol=1e-4)
            seen.append(X.numpy())
        allX = np.concatenate(seen)
        by_workers[nw] = allX
        uniq = np.unique(np.round(allX, 6), axis=0)
        print(f"num_workers={nw}: {len(allX)} items, {len(uniq)} unique "
              f"({'OK' if len(uniq) == len(allX) else 'DUPLICATES'})")

    # the property we actually want: index i -> same spectrum regardless of workers
    same = np.allclose(by_workers[0], by_workers[4])
    print(f"index->spectrum stable across num_workers: {same}")

    # reproducibility across two passes (num_workers=0)
    dl = DataLoader(ds, batch_size=64, num_workers=0, collate_fn=numpy_collate)
    a = next(iter(dl))[0].numpy()
    b = next(iter(dl))[0].numpy()
    print(f"reproducible across passes: {np.allclose(a, b)}")
    print("dataset smoke test OK")
