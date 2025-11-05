from numpy.typing import NDArray
import numpy as np
from scipy.stats import zscore #type: ignore
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor #type: ignore
from scipy.spatial.distance import mahalanobis #type: ignore
import matplotlib.pyplot as plt

def zcore_outliers(data:NDArray, **kwargs) -> NDArray:
    z = np.abs(zscore(data))
    return (z > 3).any(axis=1)

def kmeans_outliers(data:NDArray, outlier_percentile:int=95, **kwargs) -> NDArray:
    nbrs = NearestNeighbors(n_neighbors=5).fit(data)
    dists, _ = nbrs.kneighbors(data)
    return dists.mean(axis=1) > np.percentile(dists.mean(axis=1), outlier_percentile)

def mahalanobis_distance(data:NDArray, outlier_percentile:int=95, **kwargs) -> NDArray:
    cov = np.cov(data.T)
    inv_cov = np.linalg.pinv(cov)
    center = data.mean(0)
    m = np.array([mahalanobis(x, center, inv_cov) for x in data])
    return m > np.percentile(m, outlier_percentile)

def local_outlier_factor(data:NDArray, metric='cosine', **kwargs) -> NDArray:
    return LocalOutlierFactor(metric=metric, **kwargs).fit_predict(data) == -1

def show_outliers(data:NDArray, method:str='z-score', png_name:str='', **kwargs) -> NDArray:
    # returns the indices of outliers if there are any
    outliers = np.ndarray([], dtype=bool)
    match method:
        case'z-score':
            outliers = zcore_outliers(data, **kwargs)
        case 'kmeans':
            outliers = kmeans_outliers(data, **kwargs)
        case 'mahalanobis':
            outliers = mahalanobis_distance(data, **kwargs)
        case 'lof':
            outliers = local_outlier_factor(data, **kwargs)
        case _:
            raise ValueError('Unsupported outlier method: '+method)
    if outliers.sum() == 0:
        return np.array([])
    
    plt.figure(figsize=(8, 5))
    plt.plot(data[~outliers].T, color='black', alpha=0.1)
    plt.plot(data[outliers].T, color='red', alpha=0.9)
    plt.title('Outliers using '+method)
    if png_name != '':
        plt.savefig(png_name)
    plt.show()

    return np.where(outliers)[0] 
