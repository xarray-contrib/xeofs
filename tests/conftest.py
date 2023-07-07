from typing import Dict, Optional

import numpy as np
import pytest
import warnings
import xarray as xr

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


@pytest.fixture
def mock_data_array():
    rng = np.random.default_rng(7)
    return xr.DataArray(
            rng.normal(5, 3, size=(7, 4, 3)),
            dims=('time', 'lat', 'lon'),
            coords={
                'time': xr.date_range('2001', '2007', freq='YS'),
                'lat': [30.0, 40.0, 50.0, 60.0],
                'lon': [-10.0, 0.0, 10.0]
            },
            name='t2m',
            attrs=dict(description='mock_data')
        )

@pytest.fixture
def mock_dataset(mock_data_array):
    t2m = mock_data_array
    prcp = t2m**2
    return xr.Dataset({'t2m': t2m, 'prcp': prcp})

@pytest.fixture
def mock_data_array_list(mock_data_array):
    da1 = mock_data_array
    da2 = mock_data_array ** 2
    da3 = mock_data_array ** 3
    return [da1, da2, da3]

@pytest.fixture
def mock_data_array_isolated_nans(mock_data_array):
    invalid_data = mock_data_array.copy()
    invalid_data.loc[dict(time='2001', lat=30.0, lon=-10.0)] = np.nan
    return invalid_data

@pytest.fixture
def mock_data_array_full_dimensional_nans(mock_data_array):
    valid_data = mock_data_array.copy()
    valid_data.loc[dict(lat=30.0)] = np.nan
    valid_data.loc[dict(time='2002')] = np.nan
    return valid_data

@pytest.fixture
def mock_data_array_boundary_nans(mock_data_array):
    valid_data = mock_data_array.copy(deep=True)
    valid_data.loc[dict(lat=30.0)] = np.nan
    valid_data.loc[dict(time='2001')] = np.nan
    return valid_data


@pytest.fixture
def mock_dask_data_array(mock_data_array):
    return mock_data_array.chunk({'lon': 2, 'lat': 2, 'time': -1})
