"""Pure spectral augmentation transforms (numpy, array-in/array-out).

Two effects, applied by the simulator at different pipeline stages:
  - albedo_augment: multiplicative brightness. Used per-class on each class's
    intra-class mixture AND as the whole-pixel global-illumination scalar on the
    finished mixture.
  - glint_offset: additive sun + sky glint, wavelength-flat or shaped by
    `water_fresnel_shape`. Water only.
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


GLINT_REF_NM = 1050.0  # wavelength at which sun-glint magnitude is specified
SKY_REF_NM = 450.0     # wavelength at which sky-glint magnitude is specified


def water_fresnel_shape(
    wavelengths: NDArray,
    nk_csv: str,
    ref_nm: float = GLINT_REF_NM,
) -> NDArray[np.float64]:
    """Nadir Fresnel reflectance of liquid water, ((n-1)/(n+1))^2, normalised to 1 at ref_nm.

    `nk_csv` holds rows of (wavelength_um, n, k), e.g. the bundled Hale & Querry
    (1973) table. n falls through the SWIR, so specular reflection off water
    does too: relative to 1050 nm it is ~0.85 at 2200 nm and ~0.76 at 2400 nm.
    """
    d = np.loadtxt(nk_csv, delimiter=",")
    n = np.interp(np.asarray(wavelengths, float) / 1000.0, d[:, 0], d[:, 1])
    n_ref = np.interp(ref_nm / 1000.0, d[:, 0], d[:, 1])
    return ((n - 1) / (n + 1)) ** 2 / ((n_ref - 1) / (n_ref + 1)) ** 2


def glint_offset(
    spectra: NDArray[np.float32],
    low: float,
    high: float,
    shape: Optional[NDArray] = None,
    sky_range: tuple = (0.0, 0.0),
    sky_k_range: tuple = (1.0, 4.0),
    wavelengths: Optional[NDArray] = None,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float32]:
    """Per-spectrum additive sun + sky glint, shape(lambda) * (g + s * (lambda/450)^-k) (water).

    Specular reflection off water adds a term on top of the water-leaving
    signal (a vertical shift, distinct from albedo's multiplicative stretch).
    Both terms carry the surface's reflectance spectrum `shape`, e.g.
    `water_fresnel_shape(wavelengths)` (None = wavelength-flat). Per spectrum:
      - g ~ Uniform(low, high): direct-sun glint at GLINT_REF_NM. The solar
        spectrum cancels in reflectance, so this term is shape alone;
      - s ~ Uniform(sky_range): sky glint at SKY_REF_NM, following the skylight
        spectrum (lambda/SKY_REF_NM)^-k with k ~ Uniform(sky_k_range)
        (~1 aerosol-dominated haze .. 4 Rayleigh). Requires `wavelengths`.
        With sky_range (0, 0) no sky term is applied (and no s, k are drawn).
    Returns same shape/dtype as input.
    """
    if rng is None:
        rng = np.random.default_rng()
    arr, single = _as_2d(spectra)
    g = rng.uniform(low, high, size=arr.shape[0])
    if shape is None and sky_range[1] == 0:
        out = arr + g[:, None].astype(arr.dtype)
        return out[0] if single else out
    glint = np.broadcast_to(g[:, None], arr.shape)
    if sky_range[1] != 0:
        if wavelengths is None:
            raise ValueError("sky glint requires wavelengths")
        k = rng.uniform(sky_k_range[0], sky_k_range[1], size=arr.shape[0])
        s = rng.uniform(sky_range[0], sky_range[1], size=arr.shape[0])
        rel = np.asarray(wavelengths, float) / SKY_REF_NM
        glint = glint + s[:, None] * rel[None, :] ** -k[:, None]
    if shape is not None:
        glint = glint * np.asarray(shape, float)[None, :]
    out = (arr + glint).astype(arr.dtype)
    return out[0] if single else out
