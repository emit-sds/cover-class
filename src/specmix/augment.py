"""Pure spectral augmentation transforms (numpy, array-in/array-out).

Two effects, applied by the simulator at different pipeline stages:
  - albedo_augment: multiplicative brightness. Used per-class on each class's
    intra-class mixture AND as the whole-pixel global-illumination scalar on the
    finished mixture.
  - glint_offset: additive offset, wavelength-flat or shaped by
    `water_fresnel_shape`. Water only (sun/sky glint).
"""

import os
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


WATER_NK_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                            "water_nk_hale_querry_1973.csv")


def water_fresnel_shape(wavelengths: NDArray, ref_nm: float = 1050.0) -> NDArray[np.float64]:
    """Nadir Fresnel reflectance of liquid water ((n-1)/(n+1))^2, normalised to 1 at ref_nm.

    n from Hale & Querry (1973). It falls through the SWIR (1.327 at 1.0 um,
    1.279 at 2.4 um), so specular reflection off water does too: 0.85 at
    2200 nm and 0.76 at 2400 nm relative to 1050 nm.
    """
    d = np.loadtxt(WATER_NK_CSV, delimiter=",")
    n = np.interp(np.asarray(wavelengths, float) / 1000.0, d[:, 0], d[:, 1])
    n_ref = np.interp(ref_nm / 1000.0, d[:, 0], d[:, 1])
    return ((n - 1) / (n + 1)) ** 2 / ((n_ref - 1) / (n_ref + 1)) ** 2


def glint_offset(
    spectra: NDArray[np.float32],
    low: float,
    high: float,
    shape: Optional[NDArray] = None,
    tilt_range: tuple = (0.0, 0.0),
    wavelengths: Optional[NDArray] = None,
    ref_nm: float = 1050.0,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float32]:
    """Per-spectrum additive glint g * shape(lambda) [* (lambda/ref_nm)^-k] (water).

    Sun/sky specular reflection off water adds a term on top of the
    water-leaving signal. g ~ Uniform(low, high) per spectrum is its value at
    ref_nm (vertical shift, distinct from albedo's multiplicative stretch).

    shape=None adds g flat across every band (the original behaviour). Pass
    `water_fresnel_shape(wavelengths)` for the physical spectral form. A
    non-zero `tilt_range` additionally draws k ~ Uniform(tilt_range) per
    spectrum for a sky-glint tilt (needs `wavelengths`); with (0, 0) no k is
    drawn, so the RNG stream matches the flat case. Returns same shape/dtype
    as input.
    """
    if rng is None:
        rng = np.random.default_rng()
    arr, single = _as_2d(spectra)
    offsets = rng.uniform(low, high, size=arr.shape[0]).astype(arr.dtype)
    if shape is None and tilt_range[1] == 0:
        out = arr + offsets[:, None]
        return out[0] if single else out
    prof = np.ones((arr.shape[0], arr.shape[1])) if shape is None else np.tile(shape, (arr.shape[0], 1))
    if tilt_range[1] != 0:
        if wavelengths is None:
            raise ValueError("glint tilt needs wavelengths")
        k = rng.uniform(tilt_range[0], tilt_range[1], size=arr.shape[0])
        prof = prof * (np.asarray(wavelengths, float) / ref_nm)[None, :] ** -k[:, None]
    out = (arr + offsets[:, None] * prof).astype(arr.dtype)
    return out[0] if single else out
