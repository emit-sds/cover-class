import numpy as np
from typing import List
from pathlib import Path

BINARY_NO_DATA_VAL = 255
PROBA_NO_DATA_VAL = -9999

numpy_to_gdal = {
    np.dtype(np.float64): 7,
    np.dtype(np.float32): 6,
    np.dtype(np.int32): 5,
    np.dtype(np.uint32): 4,
    np.dtype(np.int16): 3,
    np.dtype(np.uint16): 2,
    np.dtype(np.uint8): 1,
}

def make_geotiff(probabilities: np.ndarray, dataset_shape: tuple, outfp: str, proba: bool, thresholds: List[float]):
    try:
        from osgeo import gdal
    except Exception as e:
        raise ImportError("GDAL (osgeo) is required for make_geotiff(). Install it (e.g., conda-forge: gdal).") from e

    nan_mask = np.isnan(probabilities)
    
    mem_driver, tiff_driver = gdal.GetDriverByName('MEM'), gdal.GetDriverByName('GTiff')
    opts = ['COMPRESS=LZW', 'COPY_SRC_OVERVIEWS=YES', 'TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256']

    if proba:
        old_shape = probabilities.shape
        probabilities[nan_mask] = PROBA_NO_DATA_VAL
        probabilities = probabilities.reshape((dataset_shape[0], dataset_shape[1], 1))
        ds = mem_driver.Create('', probabilities.shape[1], probabilities.shape[0], probabilities.shape[2], numpy_to_gdal[probabilities.dtype])
        ds.GetRasterBand(1).WriteArray(probabilities[:,:,0])    
        ds.GetRasterBand(1).SetNoDataValue(PROBA_NO_DATA_VAL)
        
        _op = Path(outfp)
        _op_s = str(_op.with_name(f"{_op.stem}_prob{_op.suffix}"))
        _ = tiff_driver.CreateCopy(_op_s, ds, options=opts)

        probabilities = probabilities.reshape(old_shape)

    binary_prob = (probabilities >= np.asarray(thresholds, dtype=probabilities.dtype)).astype(np.uint8)
    binary_prob[nan_mask] = np.uint8(BINARY_NO_DATA_VAL)

    # Reshape into input shape
    binary_prob = binary_prob.reshape((dataset_shape[0], dataset_shape[1], 1))

    ds = mem_driver.Create('', binary_prob.shape[1], binary_prob.shape[0], binary_prob.shape[2], numpy_to_gdal[binary_prob.dtype])
    ds.GetRasterBand(1).WriteArray(binary_prob[:,:,0])
    ds.GetRasterBand(1).SetNoDataValue(BINARY_NO_DATA_VAL)

    _ = tiff_driver.CreateCopy(outfp, ds, options=opts)