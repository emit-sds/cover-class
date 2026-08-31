"""Pure spectral augmentation transforms (numpy, array-in/array-out).

Two effects, applied by the simulator at different pipeline stages:
  - albedo_augment: multiplicative brightness. Used per-class on each class's
    intra-class mixture AND as the whole-pixel global-illumination scalar on the
    finished mixture.
  - glint_offset: additive wavelength-flat offset. Water only (sun/sky glint).
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def _as_2d(spectra: NDArray) -> tuple:
    arr = np.asarray(spectra)
    single = arr.ndim == 1
    return (arr[None, :] if single else arr), single


def albedo_augment(
    spectra: NDArray[np.float32],
    low: float,
    high: float,
    magnitude_max: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float32]:
    """Per-spectrum uniform-random brightness multiplier.

    Draw a factor ~ Uniform(low, high) per spectrum and multiply. If
    `magnitude_max` is set and a factor would push the spectrum's brightest band
    above it, the factor is capped so that band lands exactly at `magnitude_max`.
    This caps the MULTIPLIER, not the spectrum -- shape is preserved, nothing is
    clipped band-wise. Returns same shape/dtype as input.
    """
    if rng is None:
        rng = np.random.default_rng()
    arr, single = _as_2d(spectra)
    factors = rng.uniform(low, high, size=arr.shape[0])

    if magnitude_max is not None:
        peak = np.nanmax(arr, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            max_factor = np.where(peak > 0, magnitude_max / peak, np.inf)
        factors = np.minimum(factors, max_factor)

    out = (arr * factors[:, None]).astype(arr.dtype)
    return out[0] if single else out


def glint_offset(
    spectra: NDArray[np.float32],
    low: float,
    high: float,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float32]:
    """Per-spectrum constant offset added across all bands (water glint).

    Sun/sky specular reflection off water adds a roughly wavelength-flat term on
    top of the water-leaving signal. Model it as one constant ~ Uniform(low,
    high) per spectrum, added to every band (vertical shift, distinct from
    albedo's multiplicative stretch). Returns same shape/dtype as input.
    """
    if rng is None:
        rng = np.random.default_rng()
    arr, single = _as_2d(spectra)
    offsets = rng.uniform(low, high, size=arr.shape[0]).astype(arr.dtype)
    out = arr + offsets[:, None]
    return out[0] if single else out
