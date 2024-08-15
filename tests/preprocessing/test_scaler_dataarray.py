import pytest
import xarray as xr
import numpy as np

from xeofs.preprocessing.scaler import Scaler


@pytest.mark.parametrize(
    "with_std, with_coslat",
    [
        (True, True),
        (True, True),
        (True, False),
        (True, False),
        (False, True),
        (False, True),
        (False, False),
        (False, False),
        (True, True),
        (True, True),
        (True, False),
        (True, False),
        (False, True),
        (False, True),
        (False, False),
        (False, False),
    ],
)
def test_init_params(with_std, with_coslat):
    s = Scaler(with_std=with_std, with_coslat=with_coslat)
    assert s.get_params()["with_std"] == with_std
    assert s.get_params()["with_coslat"] == with_coslat


@pytest.mark.parametrize(
    "with_std, with_coslat",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_fit_params(with_std, with_coslat, mock_data_array):
    s = Scaler(with_std=with_std, with_coslat=with_coslat)
    sample_dims = ["time"]
    feature_dims = ["lat", "lon"]
    size_lats = mock_data_array.lat.size
    weights = xr.DataArray(np.random.rand(size_lats), dims=["lat"])
    s.fit(mock_data_array, sample_dims, feature_dims, weights)
    assert hasattr(s, "mean_"), "Scaler has no mean attribute."
    if with_std:
        assert hasattr(s, "std_"), "Scaler has no std attribute."
    if with_coslat:
        assert hasattr(s, "coslat_weights_"), "Scaler has no coslat_weights attribute."
    assert s.mean_ is not None, "Scaler mean is None."
    if with_std:
        assert s.std_ is not None, "Scaler std is None."
    if with_coslat:
        assert s.coslat_weights_ is not None, "Scaler coslat_weights is None."


@pytest.mark.parametrize(
    "with_std, with_coslat, with_weights",
    [
        (True, True, True),
        (True, False, True),
        (False, True, True),
        (False, False, True),
        (True, True, False),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_transform_params(with_std, with_coslat, with_weights, mock_data_array):
    s = Scaler(with_std=with_std, with_coslat=with_coslat)
    sample_dims = ["time"]
    feature_dims = ["lat", "lon"]
    size_lats = mock_data_array.lat.size
    if with_weights:
        weights = xr.DataArray(
            np.random.rand(size_lats), dims=["lat"], coords={"lat": mock_data_array.lat}
        )
    else:
        weights = None
    s.fit(mock_data_array, sample_dims, feature_dims, weights)
    transformed = s.transform(mock_data_array)
    assert transformed is not None, "Transformed data is None."

    transformed_mean = transformed.mean(sample_dims, skipna=False)
    assert np.allclose(transformed_mean, 0), "Mean of the transformed data is not zero."

    if with_std and not (with_coslat or with_weights):
        transformed_std = transformed.std(sample_dims, skipna=False)

        assert np.allclose(
            transformed_std, 1
        ), "Standard deviation of the transformed data is not one."

    if with_coslat:
        assert s.coslat_weights_ is not None, "Scaler coslat_weights is None."
        assert not np.array_equal(
            transformed, mock_data_array
        ), "Data has not been transformed."

    transformed2 = s.fit_transform(mock_data_array, sample_dims, feature_dims, weights)
    xr.testing.assert_allclose(transformed, transformed2)


@pytest.mark.parametrize(
    "with_std, with_coslat",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_inverse_transform_params(with_std, with_coslat, mock_data_array):
    s = Scaler(
        with_std=with_std,
        with_coslat=with_coslat,
    )
    sample_dims = ["time"]
    feature_dims = ["lat", "lon"]
    size_lats = mock_data_array.lat.size
    weights = xr.DataArray(
        np.random.rand(size_lats), dims=["lat"], coords={"lat": mock_data_array.lat}
    )
    s.fit(mock_data_array, sample_dims, feature_dims, weights)
    transformed = s.transform(mock_data_array)
    inverted = s.inverse_transform_data(transformed)
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
    s = Scaler(with_std=True)
    s.fit(mock_data_array, dim_sample, dim_feature)
    assert hasattr(s, "mean_"), "Scaler has no mean attribute."
    assert s.mean_ is not None, "Scaler mean is None."
    assert hasattr(s, "std_"), "Scaler has no std attribute."
    assert s.std_ is not None, "Scaler std is None."
    # check that all dimensions are present except the sample dimensions
    assert set(s.mean_.dims) == set(mock_data_array.dims) - set(
        dim_sample
    ), "Mean has wrong dimensions."
    assert set(s.std_.dims) == set(mock_data_array.dims) - set(
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
    s = Scaler()
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
    s = Scaler()

    with pytest.raises(TypeError):
        s.fit(mock_data_array_list, ["time"], ["lon", "lat"])

    s.fit(mock_data_array, ["time"], ["lon", "lat"])

    with pytest.raises(TypeError):
        s.transform(mock_data_array_list)
