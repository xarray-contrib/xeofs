import pytest
import xarray as xr
import numpy as np

from xeofs.models.scaler import Scaler, ListScaler

# TEST INPUT TYPES
def test_fit_input_type():
    s = Scaler()
    with pytest.raises(TypeError):
        s.fit("not a DataArray", ['time'], ['lon', 'lat'])  # type: ignore

def test_transform_input_type():
    s = Scaler()
    with pytest.raises(TypeError):
        s.transform("not a DataArray")  # type: ignore

def test_inverse_transform_input_type():
    s = Scaler()
    with pytest.raises(TypeError):
        s.inverse_transform("not a DataArray")  # type: ignore

def test_compute_coslat_weights_input_type():
    s = Scaler()
    with pytest.raises(TypeError):
        s._compute_sqrt_cos_lat_weights("not a DataArray", ['lon', 'lat'])  # type: ignore

def test_list_scaler_input_type():
    s = ListScaler()
    with pytest.raises(TypeError):
        s.fit('not a list of DataArrays', ['time'], ['lon'])  # type: ignore

# TEST TRANSFORMATIONS

@pytest.fixture
def create_data():
    """Create a 2D xarray DataArray with latitudes as one of the dimensions."""
    lats = np.array([10, 20, 30, 40], dtype=np.float64)
    coords_lat = xr.DataArray(lats, dims=['lat'], coords={'lat': lats})
    da = xr.DataArray([[1., 2, 3, 4], [5, 6, 7, 8], [3, 4, 2, 1]], dims=['time', 'lat'], coords={'time': [1, 2, 3], 'lat': coords_lat})
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
    scaler = Scaler(with_std=False, with_coslat=False)
    scaler.fit(create_data, ['time'], ['lat'])
    transformed = scaler.transform(create_data)
    np.testing.assert_array_almost_equal(transformed.mean('time').values, np.zeros(transformed.shape[1]))  # type: ignore

def test_std_scaling(create_data):
    '''Test that standard deviation is set to 1.'''
    scaler = Scaler(with_std=True, with_coslat=False)
    scaler.fit(create_data, ['time'], ['lat'])
    transformed = scaler.transform(create_data)
    np.testing.assert_array_almost_equal(transformed.std('time').values, np.ones(transformed.shape[1]))  # type: ignore

def test_coslat_scaling(create_data):
    '''Test that coslat weights are applied.'''
    scaler = Scaler(with_std=False, with_coslat=True)
    scaler.fit(create_data, ['time'], ['lat'])
    transformed = scaler.transform(create_data)
    # For simplicity, check only that coslat_weights are not None and are applied
    assert scaler.coslat_weights is not None
    assert not np.array_equal(transformed, create_data)

def test_weights(create_data, create_weights):
    '''Test that weights are applied.'''
    weights = create_weights
    scaler = Scaler(with_std=False, with_coslat=False, with_weights=True)
    scaler.fit(create_data, ['time'], ['lat'], weights)
    transformed = scaler.transform(create_data)
    # For simplicity, check only that weights are not None and are applied
    assert scaler.weights is not None
    assert not np.array_equal(transformed, create_data)

def test_inverse_transform(create_data, create_weights):
    '''Test that inverse transform is the inverse of transform.'''
    weights = create_weights
    scaler = Scaler(with_std=True, with_coslat=True, with_weights=True)
    scaler.fit(create_data, ['time'], ['lat'], weights)
    transformed = scaler.transform(create_data)
    inverted = scaler.inverse_transform(transformed)
    np.testing.assert_array_almost_equal(inverted, create_data)

def test_list_scaler(create_data):
    '''Test that list scaler works.'''
    data_list = [create_data, create_data.isel(lat=slice(0, 2))]
    scaler = ListScaler(with_std=True, with_coslat=False)
    scaler.fit(data_list, ['time'], [['lat'], ['lat']])
    transformed = scaler.transform(data_list)
    # test that transformed data has zero mean and unit std
    for trans in transformed:
        np.testing.assert_array_almost_equal(trans.mean('time').values, np.zeros(trans.shape[1]))
        np.testing.assert_array_almost_equal(trans.std('time').values, np.ones(trans.shape[1]))
    inverted = scaler.inverse_transform(transformed)
    for inv, data in zip(inverted, data_list):
        np.testing.assert_array_almost_equal(inv, data)
