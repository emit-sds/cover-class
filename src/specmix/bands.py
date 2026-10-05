"""Bad-band dropping for EMIT spectra (285-band raw grid -> 207 good bands).

The two middle ranges cover strong atmospheric water-vapor absorption features
where reflectance retrievals are unreliable; the first and last trim noisy
detector edges (blue, far-SWIR). Applying these to the 285-band endmember
library yields the 207 "good" bands used in the labeled validation set.
"""

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

DROP_WL_RANGES: List[List[int]] = [
    [0, 440],
    [1310, 1490],
    [1770, 2050],
    [2440, 2880],
]


def good_band_mask(
    banddef: NDArray,
    drop_wl_ranges: Optional[List[List[int]]] = None,
) -> NDArray[np.bool_]:
    """Boolean mask (True = keep) over `banddef` for the given drop ranges."""
    mask = np.ones_like(banddef, dtype=bool)
    if not drop_wl_ranges:
        return mask
    for low, high in drop_wl_ranges:
        mask ^= (banddef >= low) & (banddef <= high)
    return mask


def drop_bad_bands(
    data_matrix: NDArray[np.float32],
    banddef: NDArray,
    drop_wl_ranges: Optional[List[List[int]]] = None,
) -> NDArray[np.float32]:
    """Remove high-uncertainty bands from a spectrum or array of spectra.

    Args:
        data_matrix: spectra with wavelength along the last axis.
        banddef: wavelength (nm) of each band, aligned to the last axis.
        drop_wl_ranges: list of [low, high] nm ranges (inclusive) to drop.

    Returns:
        The spectra with the dropped bands removed from the last axis.
    """
    if not drop_wl_ranges:
        return data_matrix
    mask = good_band_mask(banddef, drop_wl_ranges)
    return np.delete(data_matrix, ~mask, axis=-1)
