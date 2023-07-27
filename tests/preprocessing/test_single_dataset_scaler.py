import pytest
import xarray as xr
import numpy as np

from xeofs.preprocessing.scaler import SingleDatasetScaler


@pytest.mark.parametrize('with_std, with_coslat, with_weights', [
    (True, True, True),
    (True, True, False),
    (True, False, True),
    (True, False, False),
    (False, True, True),
    (False, True, False),
    (False, False, True),
    (False, False, False),
    (True, True, True),
    (True, True, False),
    (True, False, True),
    (True, False, False),
    (False, True, True),
    (False, True, False),
    (False, False, True),
    (False, False, False)
])
def test_init_params(with_std, with_coslat, with_weights):
    s = SingleDatasetScaler(with_std=with_std, with_coslat=with_coslat, with_weights=with_weights)
    assert hasattr(s, '_params')
    assert s._params['with_std'] == with_std
    assert s._params['with_coslat'] == with_coslat
    assert s._params['with_weights'] == with_weights


@pytest.mark.parametrize('with_std, with_coslat, with_weights', [
    (True, True, True),
    (True, True, False),
    (True, False, True),
    (True, False, False),
    (False, True, True),
    (False, True, False),
    (False, False, True),
    (False, False, False),
    (True, True, True),
    (True, True, False),
    (True, False, True),
    (True, False, False),
    (False, True, True),
    (False, True, False),
    (False, False, True),
    (False, False, False)
])
def test_fit_params(with_std, with_coslat, with_weights, mock_dataset):
    s = SingleDatasetScaler(with_std=with_std, with_coslat=with_coslat, with_weights=with_weights)
    sample_dims = ['time']
    feature_dims = ['lat', 'lon']
    size_lats = mock_dataset.lat.size
    weights = xr.DataArray(np.random.rand(size_lats), dims=['lat'], name='weights').to_dataset()
    s.fit(mock_dataset, sample_dims, feature_dims, weights)
    assert hasattr(s, 'mean'), "Scaler has no mean attribute."
    if with_std:
        assert hasattr(s, 'std'), "Scaler has no std attribute."
    if with_coslat:
        assert hasattr(s, 'coslat_weights'), "Scaler has no coslat_weights attribute."
    if with_weights:
        assert hasattr(s, 'weights'), "Scaler has no weights attribute."
    assert s.mean is not None, "Scaler mean is None."
    if with_std:
        assert s.std is not None, "Scaler std is None."
    if with_coslat:
        assert s.coslat_weights is not None, "Scaler coslat_weights is None."
    if with_weights:
        assert s.weights is not None, "Scaler weights is None."


@pytest.mark.parametrize('with_std, with_coslat, with_weights', [
    (True, True, True),
    (True, True, False),
    (True, False, True),
    (True, False, False),
    (False, True, True),
    (False, True, False),
    (False, False, True),
    (False, False, False),
    (True, True, True),
    (True, True, False),
    (True, False, True),
    (True, False, False),
    (False, True, True),
    (False, True, False),
    (False, False, True),
    (False, False, False)
])
def test_transform_params(with_std, with_coslat, with_weights, mock_dataset):
    s = SingleDatasetScaler(with_std=with_std, with_coslat=with_coslat, with_weights=with_weights)
    sample_dims = ['time']
    feature_dims = ['lat', 'lon']
    size_lats = mock_dataset.lat.size
    weights1 = xr.DataArray(np.random.rand(size_lats), dims=['lat'], name='t2m')
    weights2 = xr.DataArray(np.random.rand(size_lats), dims=['lat'], name='prcp')
    weights = xr.merge([weights1, weights2])
    s.fit(mock_dataset, sample_dims, feature_dims, weights)
    transformed = s.transform(mock_dataset)
    assert transformed is not None, "Transformed data is None."

    transformed_mean = transformed.mean(sample_dims, skipna=False)
    assert np.allclose(transformed_mean.to_array(), 0), 'Mean of the transformed data is not zero.'

    if with_std:
        transformed_std = transformed.std(sample_dims, skipna=False)
        if with_coslat or with_weights:
            assert (transformed_std <= 1).all(), 'Standard deviation of the transformed data is larger one.'
        else:
            assert np.allclose(transformed_std.to_array(), 1), 'Standard deviation of the transformed data is not one.'

    if with_coslat:
        assert s.coslat_weights is not None, "Scaler coslat_weights is None."
        assert not np.array_equal(transformed, mock_dataset), "Data has not been transformed."

    if with_weights:
        assert s.weights is not None, "Scaler weights is None."
        assert not np.array_equal(transformed, mock_dataset), "Data has not been transformed."

    transformed2 = s.fit_transform(mock_dataset, sample_dims, feature_dims, weights)
    xr.testing.assert_allclose(transformed, transformed2)


@pytest.mark.parametrize('with_std, with_coslat, with_weights', [
    (True, True, True),
    (True, True, False),
    (True, False, True),
    (True, False, False),
    (False, True, True),
    (False, True, False),
    (False, False, True),
    (False, False, False),
    (True, True, True),
    (True, True, False),
    (True, False, True),
    (True, False, False),
    (False, True, True),
    (False, True, False),
    (False, False, True),
    (False, False, False)
])
def test_inverse_transform_params(with_std, with_coslat, with_weights, mock_dataset):
    s = SingleDatasetScaler(with_std=with_std, with_coslat=with_coslat, with_weights=with_weights)
    sample_dims = ['time']
    feature_dims = ['lat', 'lon']
    size_lats = mock_dataset.lat.size
    weights1 = xr.DataArray(np.random.rand(size_lats), dims=['lat'], name='t2m')
    weights2 = xr.DataArray(np.random.rand(size_lats), dims=['lat'], name='prcp')
    weights = xr.merge([weights1, weights2])
    s.fit(mock_dataset, sample_dims, feature_dims, weights)
    transformed = s.transform(mock_dataset)
    inverted = s.inverse_transform(transformed)
    xr.testing.assert_allclose(inverted, mock_dataset)


@pytest.mark.parametrize('dim_sample, dim_feature', [
    (('time',), ('lat', 'lon')),
    (('time',), ('lon', 'lat')),
    (('lat', 'lon'), ('time',)),
    (('lon', 'lat'), ('time',)),
])
def test_fit_dims(dim_sample, dim_feature, mock_dataset):
    s = SingleDatasetScaler()
    s.fit(mock_dataset, dim_sample, dim_feature)
    assert hasattr(s, 'mean'), "Scaler has no mean attribute."
    assert s.mean is not None, "Scaler mean is None."
    assert hasattr(s, 'std'), "Scaler has no std attribute."
    assert s.std is not None, "Scaler std is None."
    # check that all dimensions are present except the sample dimensions
    assert set(s.mean.dims) == set(mock_dataset.dims) - set(dim_sample), "Mean has wrong dimensions."
    assert set(s.std.dims) == set(mock_dataset.dims) - set(dim_sample), "Standard deviation has wrong dimensions."


@pytest.mark.parametrize('dim_sample, dim_feature', [
    (('time',), ('lat', 'lon')),
    (('time',), ('lon', 'lat')),
    (('lat', 'lon'), ('time',)),
    (('lon', 'lat'), ('time',)),
])
def test_fit_transform_dims(dim_sample, dim_feature, mock_dataset):
    s = SingleDatasetScaler()
    transformed = s.fit_transform(mock_dataset, dim_sample, dim_feature)
    # check that all dimensions are present
    assert set(transformed.dims) == set(mock_dataset.dims), "Transformed data has wrong dimensions."
    # check that the coordinates are the same
    for dim in mock_dataset.dims:
        xr.testing.assert_allclose(transformed[dim], mock_dataset[dim])


# Test input types
def test_fit_input_type(mock_dataset, mock_data_array, mock_data_array_list):
    s = SingleDatasetScaler()
    # Cannot fit DataArray
    with pytest.raises(TypeError):
        s.fit(mock_data_array, ['time'], ['lon', 'lat'])
    # Cannot fit list of DataArrays
    with pytest.raises(TypeError):
        s.fit(mock_data_array_list, ['time'], ['lon', 'lat'])
    
    s.fit(mock_dataset, ['time'], ['lon', 'lat'])
    # Cannot transform DataArray
    with pytest.raises(TypeError):
        s.transform(mock_data_array)
    # Cannot transform list of DataArrays
    with pytest.raises(TypeError):
        s.transform(mock_data_array_list)


# def test_fit_weights_input_type(mock_dataset):
#     s = SingleDatasetScaler()
#     # Fitting with weights requires that the weights have the same variables as the dataset
#     # used for fitting; otherwise raise an error
#     size_lats = mock_dataset.lat.size
#     weights1 = xr.DataArray(np.random.rand(size_lats), dims=['lat'], name='t2m')  # correct name
#     weights2 = xr.DataArray(np.random.rand(size_lats), dims=['lat'], name='prcp') # correct name
#     weights3 = xr.DataArray(np.random.rand(size_lats), dims=['lat'], name='sic') # wrong name
#     weights_correct = xr.merge([weights1, weights2]) 
#     weights_wrong = xr.merge([weights1, weights3])

#     s.fit(mock_dataset, ['time'], ['lon', 'lat'], weights_correct)

#     with pytest.raises(ValueError):
#         s.fit(mock_dataset, ['time'], ['lon', 'lat'], weights_wrong)
