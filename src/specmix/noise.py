"""Sensor noise model: white floor + brightness-scaled correlated noise.

Two independent terms:

  - WHITE iid noise = signal-independent read/dark floor. Always on, fixed std,
    same on every pixel -- keeps dark pixels (water) from being noise-free.
  - CORRELATED noise ~ N(0, Σ), Σ = the measured EMIT band-covariance. The
    shot-noise-like, signal-dependent part: its magnitude scales with pixel
    brightness (mean reflectance), so a dark pixel gets almost none and a bright
    soil/snow pixel gets the full amount.

Σ is symmetrized and PD-clamped once at construction, and its Cholesky factor is
precomputed so each draw is a cheap `z @ L.T`. numpy/CPU only.
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray


class NoiseModel:
    def __init__(
        self,
        covariance: Optional[NDArray] = None,
        white_floor_std: float = 1e-4,
        covariance_scalar: float = 1.0,
        brightness_scaled: bool = True,
        min_eigenvalue: float = 1e-10,
    ):
        """
        Args:
            covariance: (B, B) band covariance Σ, or None to disable the
                correlated term (white floor only).
            white_floor_std: std of the always-on signal-independent white noise.
            covariance_scalar: gain on the correlated term (applied to std).
            brightness_scaled: if True, correlated noise magnitude scales with
                each pixel's mean reflectance (dark pixels get less).
            min_eigenvalue: floor for the PD clamp on Σ.
        """
        self.white_floor_std = float(white_floor_std)
        self.covariance_scalar = float(covariance_scalar)
        self.brightness_scaled = bool(brightness_scaled)
        self._chol = None
        if covariance is not None:
            cov = np.asarray(covariance, dtype=np.float64)
            cov = 0.5 * (cov + cov.T)  # symmetrize
            w, v = np.linalg.eigh(cov)
            w = np.clip(w, min_eigenvalue, None)  # PD clamp
            cov_pd = (v * w) @ v.T
            self._chol = np.linalg.cholesky(cov_pd)  # L, cov = L @ L.T
            self.n_bands = cov.shape[0]

    def sample(self, spectra: NDArray[np.float32],
               rng: np.random.Generator) -> NDArray[np.float32]:
        """Return an additive noise array shaped like `spectra` ((N, B) or (B,)).

        Caller adds it to the (clean) spectra. Correlated term scales per-row by
        mean brightness when enabled; white floor is uniform.
        """
        arr = np.atleast_2d(spectra)
        n, b = arr.shape

        # White floor (always on, signal-independent).
        noise = rng.normal(0.0, self.white_floor_std, size=(n, b))

        # Correlated brightness-scaled term.
        if self._chol is not None:
            z = rng.standard_normal(size=(n, b))
            corr = (z @ self._chol.T) * self.covariance_scalar
            if self.brightness_scaled:
                brightness = arr.mean(axis=1, keepdims=True)  # per-pixel mean rfl
                corr = corr * brightness
            noise = noise + corr

        noise = noise.astype(np.float32)
        return noise[0] if np.ndim(spectra) == 1 else noise
