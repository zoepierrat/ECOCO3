"""
Shared helpers for matching FLUXNET/ECOCO3 records to gridded SPI/SPEI drought
indices from the CDS ERA5-derived monthly drought dataset (data/Support/Drought/),
one NetCDF file per month per index (e.g. SPI1_..._201801.nc, SPEI1_..._201801.nc).

Used by 00a_FLUXNET_datashuttle_preprocess.ipynb, 00b_ECOCO_preprocess_C1_V3.ipynb,
and 00b_ECOCO_preprocess_C2_V1.ipynb.
"""

import glob
import os

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


def load_drought_index(folder, prefix, variable):
    """Load one-file-per-month NetCDFs matching `prefix` (e.g. 'SPI1' or
    'SPEI1') from `folder` into lookup structures for extract_drought_index.
    `variable` is the data variable name inside each file.
    """
    files = sorted(glob.glob(os.path.join(folder, f'{prefix}_*.nc')))
    if not files:
        raise FileNotFoundError(f"No '{prefix}_*.nc' files found in {folder}")

    times, arrays = [], []
    lat = lon = None
    for f in files:
        ds = xr.open_dataset(f)
        times.append(pd.Timestamp(ds['time'].values[0]))
        arrays.append(ds[variable].values[0])  # (lat, lon)
        if lat is None:
            lat = ds['lat'].values
            lon = ds['lon'].values
        ds.close()

    values = np.stack(arrays, axis=0)  # (time, lat, lon)
    lookup = {(t.year, t.month): i for i, t in enumerate(times)}

    lat2d, lon2d = np.meshgrid(lat, lon, indexing='ij')
    points = np.column_stack((lat2d.ravel(), lon2d.ravel()))
    tree = cKDTree(points)

    return {
        'lookup': lookup,
        'tree': tree,
        'values': values,
        'nlat': len(lat),
        'nlon': len(lon),
    }


def extract_drought_index(index, lat, lon, timestamp):
    """Nearest-grid-cell, same-(year, month) lookup of a drought index value."""
    timestamp = pd.to_datetime(timestamp)
    time_idx = index['lookup'].get((timestamp.year, timestamp.month))
    if time_idx is None:
        return np.nan

    _, idx = index['tree'].query((lat, lon))
    lat_idx, lon_idx = np.unravel_index(idx, (index['nlat'], index['nlon']))
    val = index['values'][time_idx, lat_idx, lon_idx]

    if np.isnan(val) or np.isinf(val):
        return np.nan
    return float(val)
