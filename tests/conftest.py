from typing import Dict, Optional

import numpy as np
import pytest
import warnings
import xarray as xr

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# =============================================================================
# Input data
# =============================================================================
@pytest.fixture
def mock_data_array():
    rng = np.random.default_rng(7)
    noise = rng.normal(5, 3, size=(25, 5, 4))
    signal = 2*np.sin(np.linspace(0, 2*np.pi, 25))[:, None, None]
    return xr.DataArray(
            signal + noise,
            dims=('time', 'lat', 'lon'),
            coords={
                'time': xr.date_range('2001', '2025', freq='YS'),
                'lat': [20.0, 30.0, 40.0, 50.0, 60.0],
                'lon': [-10.0, 0.0, 10.0, 20.0]
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


# =============================================================================
# Intermediate data
# =============================================================================
@pytest.fixture
def sample_input_data():
    '''Create a sample input data.'''
    return xr.DataArray(np.random.rand(10, 20), dims=('sample', 'feature'))

@pytest.fixture
def sample_components():
    '''Create a sample components.'''
    return xr.DataArray(np.random.rand(20, 5), dims=('feature', 'mode'))

@pytest.fixture
def sample_scores():
    '''Create a sample scores.'''
    return xr.DataArray(np.random.rand(10, 5), dims=('sample', 'mode'))

@pytest.fixture
def sample_exp_var():
    return xr.DataArray(
        np.random.rand(10),
        dims=('mode',),
        coords={'mode': np.arange(10)},
        name='explained_variance'
    )

@pytest.fixture
def sample_total_variance(sample_exp_var):
    return sample_exp_var.sum()

@pytest.fixture
def sample_idx_modes_sorted(sample_exp_var):
    return sample_exp_var.argsort()[::-1]

@pytest.fixture
def sample_norm():
    return xr.DataArray(np.random.rand(10), dims=('mode',))

@pytest.fixture
def sample_squared_covariance():
    return xr.DataArray(np.random.rand(10), dims=('mode',))

@pytest.fixture
def sample_total_squared_covariance(sample_squared_covariance):
    return sample_squared_covariance.sum('mode')

@pytest.fixture
def sample_rotation_matrix():
    '''Create a sample rotation matrix.'''
    return xr.DataArray(np.random.rand(5, 5), dims=('mode_m', 'mode_n'))

@pytest.fixture
def sample_phi_matrix():
    '''Create a sample phi matrix.'''
    return xr.DataArray(np.random.rand(5, 5), dims=('mode_m', 'mode_n'))

@pytest.fixture
def sample_modes_sign():
    '''Create a sample modes sign.'''
    return xr.DataArray(np.random.choice([-1, 1], size=5), dims=('mode',))
