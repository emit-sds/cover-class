from collections import defaultdict
from typing import Tuple
from torch import FloatTensor
import torch
from numpy.typing import NDArray
import numpy as np
from scipy.spatial import ConvexHull # type: ignore[import]
from sklearn_extra.cluster import KMedoids # type: ignore[import]
from sklearn.cluster import KMeans # type: ignore[import]
from sklearn.decomposition import PCA # type: ignore[import]
from scipy.spatial.distance import mahalanobis # type: ignore[import]

'''
All functions in this file are meant to be used on a per-class basis
'''

def convex_hull(data_matrix: NDArray[np.float32], num_pc:int, n_samples:int, **kwargs) -> FloatTensor:
    Z_c, _ = pca(data_matrix, num_pc)

    hv = ConvexHull(Z_c, **kwargs).vertices
    if n_samples >= hv.size:
        return FloatTensor(torch.from_numpy(data_matrix[hv]).to(torch.float32))

    # Greedy farthest point sampling in the set of hull vertices
    V = Z_c[hv]
    i = np.argmax(np.einsum('ij,ij->i', V, V)) # arbitrary starting point (max squared magnitude)
    sel = [i]
    for _ in range(1, min(n_samples, len(V))):
        d2 = np.sum((V - V[i])**2, axis=1) # PCA in Euclidean space
        d2[np.array(sel)] = -np.inf
        i = np.argmax(d2) # Get furthest point from base point
        sel.append(i)
        
    idx = hv[np.array(sel)]
    return FloatTensor(torch.from_numpy(data_matrix[idx]).to(torch.float32))


def kmedoids(data_matrix: NDArray[np.float32], num_pc:int, n_samples:int, **kwargs) -> FloatTensor:
    ''' NOTE: this is only for Euclidean distances '''
    n_samples = min(len(data_matrix), n_samples)
    pca = PCA(n_components=num_pc, svd_solver="arpack", random_state=0)
    Z_c = pca.fit_transform(data_matrix)
    if kwargs.get("metric", None) == "mahalanobis":
        VI = np.linalg.inv(np.cov(Z_c, rowvar=False))
        def maha(u, v, VI=VI): 
            return mahalanobis(u, v, VI)
        kwargs["metric"] = maha
    centroids_idx = KMedoids(n_clusters=n_samples, **kwargs).fit(Z_c).medoid_indices_
    return FloatTensor(torch.from_numpy(data_matrix[centroids_idx]).to(torch.float32))


def kmeans(data_matrix: NDArray[np.float32], num_pc:int, n_samples:int, **kwargs) -> FloatTensor:
    ''' NOTE: this is only for Euclidean distances '''
    n_samples = min(len(data_matrix), n_samples)
    pca = PCA(n_components=num_pc, svd_solver="arpack", random_state=0)
    Z_c = pca.fit_transform(data_matrix)
    centroids_pca = KMeans(n_clusters=n_samples, **kwargs).fit(Z_c).cluster_centers_
    centroids_spectra = pca.inverse_transform(centroids_pca)
    return FloatTensor(torch.from_numpy(centroids_spectra).to(torch.float32))


def lhs(data_matrix: NDArray[np.float32], num_pc:int, hypercubes_per_dimension:int, samples_per_hypercube:int) -> FloatTensor:
    Z_c, _ = pca(data_matrix, num_pc)

    ## Get the min and max bounds for each PC dimension
    mins, maxs = Z_c.min(axis=0), Z_c.max(axis=0)

    ## Get the widths of each sub-hypercube to sample from
    width = (maxs - mins) / hypercubes_per_dimension
    width[width == 0] = 1.0 # Avoid division by zero if a principal component is constant

    ## Get the indices of each sub-hypercube that each PC dimension of Z falls into
    idx = np.floor((Z_c - mins)/ width).astype(int)
    idx = np.clip(idx, 0, hypercubes_per_dimension - 1)

    ## Make a dictionary to get how many points are in each sub-hypercube
    buckets = defaultdict(list)
    for local_idx, cube_id in enumerate(map(tuple, idx)):
        buckets[cube_id].append(local_idx)

    ## Sample the DPs
    chosen_local: list[int] = []
    for cube_id, pts in buckets.items():
        if len(pts) <= samples_per_hypercube:
            chosen_local.extend(pts)
        else:
            chosen_local.extend(np.random.choice(pts, size=samples_per_hypercube, replace=False))

    subsampled_data = data_matrix[np.array(chosen_local)]
    return FloatTensor(torch.from_numpy(subsampled_data).to(torch.float32)) 


def pca(X: np.ndarray, num_pc: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=num_pc, svd_solver="arpack", random_state=0)
    Z = pca.fit_transform(X)
    return Z, pca.explained_variance_ratio_
