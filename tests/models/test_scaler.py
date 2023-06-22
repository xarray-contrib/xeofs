import pytest
import xarray as xr
import numpy as np

from xeofs.models.scaler import Scaler, ListScaler

@pytest.fixture
def create_data():
    """Create a 2D xarray DataArray with latitudes as one of the dimensions."""
    lats = np.array([10, 20, 30, 40], dtype=np.float64)
    coords_lat = xr.DataArray(lats, dims=['lat'], coords={'lat': lats})
    da = xr.DataArray([[1., 2, 3, 4], [5, 6, 7, 8]], dims=['time', 'lat'], coords={'time': [1, 2], 'lat': coords_lat})
    return da

@pytest.fixture
def create_weights():
    """Create a 2D xarray DataArray with latitudes as one of the dimensions."""
    lats = np.array([10, 20, 30, 40], dtype=np.float64)
    wgths = np.array([.5, .3, .2, .7], dtype=np.float64)
    wghts = xr.DataArray(wgths, dims=['lat'], coords={'lat': lats})
    return wghts

def test_mean_scaling(create_data):
    '''Test that mean is removed from data.'''
    scaler = Scaler(['time'], ['lat'], with_mean=True, with_std=False, with_coslat=False)
    scaler.fit(create_data)
    transformed = scaler.transform(create_data)
    np.testing.assert_array_almost_equal(transformed.mean('time').values, np.zeros(transformed.shape[1]))

def test_std_scaling(create_data):
    '''Test that standard deviation is set to 1.'''
    scaler = Scaler(['time'], ['lat'], with_mean=False, with_std=True, with_coslat=False)
    scaler.fit(create_data)
    transformed = scaler.transform(create_data)
    np.testing.assert_array_almost_equal(transformed.std('time').values, np.ones(transformed.shape[1]))

def test_coslat_scaling(create_data):
    '''Test that coslat weights are applied.'''
    scaler = Scaler(['time'], ['lat'], with_mean=False, with_std=False, with_coslat=True)
    scaler.fit(create_data)
    transformed = scaler.transform(create_data)
    # For simplicity, check only that coslat_weights are not None and are applied
    assert scaler.coslat_weights is not None
    assert not np.array_equal(transformed, create_data)

def test_weights(create_data, create_weights):
    '''Test that weights are applied.'''
    scaler = Scaler(['time'], ['lat'], with_mean=False, with_std=False, with_coslat=False, weights=create_weights)
    scaler.fit(create_data)
    transformed = scaler.transform(create_data)
    # For simplicity, check only that weights are not None and are applied
    assert scaler.weights is not None
    assert not np.array_equal(transformed, create_data)

def test_inverse_transform(create_data, create_weights):
    '''Test that inverse transform is the inverse of transform.'''
    scaler = Scaler(['time'], ['lat'], with_mean=True, with_std=True, with_coslat=True, weights=create_weights)
    scaler.fit(create_data)
    transformed = scaler.transform(create_data)
    inverted = scaler.inverse_transform(transformed)
    np.testing.assert_array_almost_equal(inverted, create_data)

def test_list_scaler(create_data):
    '''Test that list scaler works.'''
    data_list = [create_data, create_data.isel(lat=slice(0, 2))]
    scaler = ListScaler(['time'], ['lat'], with_mean=True, with_std=True, with_coslat=True)
    scaler.fit(data_list)
    transformed = scaler.transform(data_list)
    inverted = scaler.inverse_transform(transformed)
    for inv, data in zip(inverted, data_list):
        np.testing.assert_array_almost_equal(inv, data)
