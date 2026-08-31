"""Endmember subsampling — index-returning, numpy/CPU.

Cuts a class's (heavily imbalanced) endmember library down to a small diverse
working set before mixing. Each sampler returns ROW INDICES into the input array
(the simulator holds the spectra and per-row source labels).

Four methods, dispatched by name via `subsample()`:
  - random   : uniform without replacement. Baseline.
  - convex   : convex-hull vertices in PCA space + greedy farthest-point trim.
               Boundary-only; kept for comparison, not recommended as primary.
  - kmedoids : k-medoids (FasterPAM) in PCA space. Real points, spatially diverse.
  - v2       : source-stratified k-medoids (default) -- a fixed quota per source
               library, k-medoids within each source. Fixes the library mix
               explicitly instead of letting PCA geometry over-weight sprawling
               outlier libraries.
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from kmedoids import KMedoids


def _pca(x: NDArray, num_pc: int) -> NDArray:
    n_pc = min(num_pc, x.shape[0], x.shape[1])
    return PCA(n_components=n_pc, svd_solver="arpack", random_state=0).fit_transform(x)


def random_sample(em: NDArray, n_samples: int, rng: np.random.Generator) -> NDArray:
    n = min(n_samples, len(em))
    return np.sort(rng.choice(len(em), size=n, replace=False))


def convex_sample(em: NDArray, n_samples: int, num_pc: int = 4,
                  rng: Optional[np.random.Generator] = None, **_) -> NDArray:
    """Convex-hull vertices in PCA space, greedy farthest-point-trimmed to n."""
    z = _pca(em, num_pc)
    hv = ConvexHull(z).vertices
    if n_samples >= hv.size:
        return np.sort(hv)
    # Greedy farthest-point sampling among hull vertices (Euclidean in PCA space).
    v = z[hv]
    i = int(np.argmax(np.einsum("ij,ij->i", v, v)))  # max-magnitude start
    sel = [i]
    for _ in range(1, n_samples):
        d2 = np.sum((v - v[i]) ** 2, axis=1)
        d2[sel] = -np.inf
        i = int(np.argmax(d2))
        sel.append(i)
    return np.sort(hv[np.array(sel)])


def kmedoids_sample(em: NDArray, n_samples: int, num_pc: int = 4,
                    method: str = "fasterpam", metric: str = "euclidean",
                    rng: Optional[np.random.Generator] = None, **_) -> NDArray:
    """K-medoids (FasterPAM) in PCA space. Returns medoid row indices."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = min(n_samples, len(em))
    z = _pca(em, num_pc)
    kwargs = {"method": method, "metric": metric}
    if metric == "mahalanobis":
        kwargs["metric"] = "mahalanobis"
        kwargs["metric_params"] = {"VI": np.linalg.inv(np.cov(z, rowvar=False))}
    seed = int(rng.integers(2**32))  # seed FasterPAM's random init for reproducibility
    idx = KMedoids(n_clusters=n, random_state=seed, **kwargs).fit(z).medoid_indices_
    return np.sort(np.asarray(idx))


def v2_sample(em: NDArray, source: NDArray, quota_per_source: int = 10,
              num_pc: int = 4, single_source_n: int = 50,
              rng: Optional[np.random.Generator] = None, **_) -> NDArray:
    """Source-stratified k-medoids: `quota_per_source` picks per source library.

    Within each source with more than the quota, k-medoids (fasterpam, its own
    PCA on just that source's rows) selects diverse real spectra spanning the
    source's internal spread. Sources with <= quota are taken whole.

    SINGLE-SOURCE classes (e.g. water = gloria2022 only) have no cross-library
    imbalance to correct, so stratification degenerates -- per the locked v2
    decision (AGENTS.md) they fall back to a plain RANDOM draw of `single_source_n`
    (default 50), which represents the common form honestly rather than picking
    only `quota_per_source` medoids.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if len(np.unique(source)) <= 1:
        return random_sample(em, single_source_n, rng)

    chosen = []
    for lab in np.unique(source):
        idx_src = np.flatnonzero(source == lab)
        if len(idx_src) <= quota_per_source:
            chosen.extend(idx_src.tolist())
            continue
        local = kmedoids_sample(em[idx_src], quota_per_source, num_pc=num_pc, rng=rng)
        chosen.extend(idx_src[local].tolist())
    return np.array(sorted(chosen))


def subsample(method: str, em: NDArray, source: NDArray, cfg: dict,
              rng: np.random.Generator) -> NDArray:
    """Dispatch to the named method. Returns sorted row indices into `em`.

    `cfg` is the per-method sub-block from the master config (only the selected
    method's block is read). `source` (per-row library id) is used only by v2.
    """
    cfg = dict(cfg or {})
    if method == "random":
        return random_sample(em, cfg.get("n_samples", 50), rng)
    if method == "convex":
        return convex_sample(em, rng=rng, **cfg)
    if method == "kmedoids":
        return kmedoids_sample(em, rng=rng, **cfg)
    if method == "v2":
        return v2_sample(em, source, rng=rng, **cfg)
    raise ValueError(f"unknown subsample method: {method!r} "
                     f"(expected random|convex|kmedoids|v2)")
