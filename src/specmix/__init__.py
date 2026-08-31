"""specmix — a spectral mixing simulator for EMIT fractional cover.

Simulate realistic EMIT hyperspectral mixtures from pure-endmember libraries:
sample a class-presence pattern, mix within and between classes, apply per-class
brightness/glint augmentation, and add instrument noise. The result is one
(spectra, fractions) pair per call — fractions are the regression target;
binarizing to presence is the caller's choice.

Quick start
-----------
    from specmix import MixtureSimulator, MixtureDataset

    sim = MixtureSimulator()                 # packaged default config
    spectra, fractions = sim.simulate(rng)   # rng = np.random.default_rng(...)

    ds = MixtureDataset(epoch_size=100_000)  # map-style, torch DataLoader-ready

Endmember HDF5s are external (not shipped with the package). They are resolved
against a data root: pass `data_root=...`, set $SPECMIX_DATA_ROOT, or run from a
directory that contains `datasets/` (the repo root). The default config, the
mixture prior, and the noise covariance ARE bundled with the package.
"""

from .simulator import MixtureSimulator, DEFAULT_CONFIG
from .dataset import MixtureDataset, numpy_collate, fractions_to_presence
from .mixture_priors import MixturePrior, load_mixture_prior
from .bands import drop_bad_bands, good_band_mask, DROP_WL_RANGES
from .augment import albedo_augment, glint_offset
from .noise import NoiseModel
from .subsample import subsample

__all__ = [
    "MixtureSimulator",
    "MixtureDataset",
    "DEFAULT_CONFIG",
    "numpy_collate",
    "fractions_to_presence",
    "MixturePrior",
    "load_mixture_prior",
    "drop_bad_bands",
    "good_band_mask",
    "DROP_WL_RANGES",
    "albedo_augment",
    "glint_offset",
    "NoiseModel",
    "subsample",
]

__version__ = "0.1.0"
