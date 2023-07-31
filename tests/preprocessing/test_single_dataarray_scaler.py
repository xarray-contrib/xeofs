import pytest
import xarray as xr
import numpy as np

from xeofs.preprocessing.scaler import SingleDataArrayScaler


@pytest.mark.parametrize(
    "with_std, with_coslat, with_weights",
    [
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
        (False, False, False),
    ],
)
def test_init_params(with_std, with_coslat, with_weights):
    s = SingleDataArrayScaler(
        with_std=with_std, with_coslat=with_coslat, with_weights=with_weights
    )
    assert hasattr(s, "_params")
    assert s._params["with_std"] == with_std
    assert s._params["with_coslat"] == with_coslat
    assert s._params["with_weights"] == with_weights


@pytest.mark.parametrize(
    "with_std, with_coslat, with_weights",
    [
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
        (False, False, False),
    ],
)
def test_fit_params(with_std, with_coslat, with_weights, mock_data_array):
    s = SingleDataArrayScaler(
        with_std=with_std, with_coslat=with_coslat, with_weights=with_weights
    )
    sample_dims = ["time"]
    feature_dims = ["lat", "lon"]
    size_lats = mock_data_array.lat.size
    weights = xr.DataArray(np.random.rand(size_lats), dims=["lat"])
    s.fit(mock_data_array, sample_dims, feature_dims, weights)
    assert hasattr(s, "mean"), "Scaler has no mean attribute."
    if with_std:
        assert hasattr(s, "std"), "Scaler has no std attribute."
    if with_coslat:
        assert hasattr(s, "coslat_weights"), "Scaler has no coslat_weights attribute."
    if with_weights:
        assert hasattr(s, "weights"), "Scaler has no weights attribute."
    assert s.mean is not None, "Scaler mean is None."
    if with_std:
        assert s.std is not None, "Scaler std is None."
    if with_coslat:
        assert s.coslat_weights is not None, "Scaler coslat_weights is None."
    if with_weights:
        assert s.weights is not None, "Scaler weights is None."


@pytest.mark.parametrize(
    "with_std, with_coslat, with_weights",
    [
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
        (False, False, False),
    ],
)
def test_transform_params(with_std, with_coslat, with_weights, mock_data_array):
    s = SingleDataArrayScaler(
        with_std=with_std, with_coslat=with_coslat, with_weights=with_weights
    )
    sample_dims = ["time"]
    feature_dims = ["lat", "lon"]
    size_lats = mock_data_array.lat.size
    weights = xr.DataArray(
        np.random.rand(size_lats), dims=["lat"], coords={"lat": mock_data_array.lat}
    )
    s.fit(mock_data_array, sample_dims, feature_dims, weights)
    transformed = s.transform(mock_data_array)
    assert transformed is not None, "Transformed data is None."

    transformed_mean = transformed.mean(sample_dims, skipna=False)
    assert np.allclose(transformed_mean, 0), "Mean of the transformed data is not zero."

    if with_std:
        transformed_std = transformed.std(sample_dims, skipna=False)
        if with_coslat or with_weights:
            assert (
                transformed_std <= 1
            ).all(), "Standard deviation of the transformed data is larger one."
        else:
            assert np.allclose(
                transformed_std, 1
            ), "Standard deviation of the transformed data is not one."

    if with_coslat:
        assert s.coslat_weights is not None, "Scaler coslat_weights is None."
        assert not np.array_equal(
            transformed, mock_data_array
        ), "Data has not been transformed."

    if with_weights:
        assert s.weights is not None, "Scaler weights is None."
        assert not np.array_equal(
            transformed, mock_data_array
        ), "Data has not been transformed."

    transformed2 = s.fit_transform(mock_data_array, sample_dims, feature_dims, weights)
    xr.testing.assert_allclose(transformed, transformed2)


@pytest.mark.parametrize(
    "with_std, with_coslat, with_weights",
    [
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
        (False, False, False),
    ],
)
def test_inverse_transform_params(with_std, with_coslat, with_weights, mock_data_array):
    s = SingleDataArrayScaler(
        with_std=with_std, with_coslat=with_coslat, with_weights=with_weights
    )
    sample_dims = ["time"]
    feature_dims = ["lat", "lon"]
    size_lats = mock_data_array.lat.size
    weights = xr.DataArray(
        np.random.rand(size_lats), dims=["lat"], coords={"lat": mock_data_array.lat}
    )
    s.fit(mock_data_array, sample_dims, feature_dims, weights)
    transformed = s.transform(mock_data_array)
    inverted = s.inverse_transform(transformed)
    xr.testing.assert_allclose(inverted, mock_data_array)


@pytest.mark.parametrize(
    "dim_sample, dim_feature",
    [
        (("time",), ("lat", "lon")),
        (("time",), ("lon", "lat")),
        (("lat", "lon"), ("time",)),
        (("lon", "lat"), ("time",)),
    ],
)
def test_fit_dims(dim_sample, dim_feature, mock_data_array):
    s = SingleDataArrayScaler()
    s.fit(mock_data_array, dim_sample, dim_feature)
    assert hasattr(s, "mean"), "Scaler has no mean attribute."
    assert s.mean is not None, "Scaler mean is None."
    assert hasattr(s, "std"), "Scaler has no std attribute."
    assert s.std is not None, "Scaler std is None."
    # check that all dimensions are present except the sample dimensions
    assert set(s.mean.dims) == set(mock_data_array.dims) - set(
        dim_sample
    ), "Mean has wrong dimensions."
    assert set(s.std.dims) == set(mock_data_array.dims) - set(
        dim_sample
    ), "Standard deviation has wrong dimensions."


@pytest.mark.parametrize(
    "dim_sample, dim_feature",
    [
        (("time",), ("lat", "lon")),
        (("time",), ("lon", "lat")),
        (("lat", "lon"), ("time",)),
        (("lon", "lat"), ("time",)),
    ],
)
def test_fit_transform_dims(dim_sample, dim_feature, mock_data_array):
    s = SingleDataArrayScaler()
    transformed = s.fit_transform(mock_data_array, dim_sample, dim_feature)
    # check that all dimensions are present
    assert set(transformed.dims) == set(
        mock_data_array.dims
    ), "Transformed data has wrong dimensions."
    # check that the coordinates are the same
    for dim in mock_data_array.dims:
        xr.testing.assert_allclose(transformed[dim], mock_data_array[dim])


# Test input types
def test_fit_input_type(mock_data_array, mock_dataset, mock_data_array_list):
    s = SingleDataArrayScaler()
    with pytest.raises(TypeError):
        s.fit(mock_dataset, ["time"], ["lon", "lat"])
    with pytest.raises(TypeError):
        s.fit(mock_data_array_list, ["time"], ["lon", "lat"])

    s.fit(mock_data_array, ["time"], ["lon", "lat"])
    with pytest.raises(TypeError):
        s.transform(mock_dataset)
    with pytest.raises(TypeError):
        s.transform(mock_data_array_list)
