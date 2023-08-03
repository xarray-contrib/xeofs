import pytest
import xarray as xr
import numpy as np

from xeofs.preprocessing.stacker import SingleDataArrayStacker


@pytest.mark.parametrize(
    "dim_sample, dim_feature",
    [
        (("time",), ("lat", "lon")),
        (("time",), ("lon", "lat")),
        (("lat", "lon"), ("time",)),
        (("lon", "lat"), ("time",)),
    ],
)
def test_fit_transform(
    dim_sample,
    dim_feature,
    mock_data_array,
    mock_data_array_isolated_nans,
    mock_data_array_full_dimensional_nans,
    mock_data_array_boundary_nans,
):
    # Test basic functionality
    stacker = SingleDataArrayStacker()
    stacked = stacker.fit_transform(mock_data_array, dim_sample, dim_feature)
    assert stacked.ndim == 2
    assert set(stacked.dims) == {"sample", "feature"}
    assert not stacked.isnull().any()

    # Test that the operation is reversible
    unstacked = stacker.inverse_transform_data(stacked)
    xr.testing.assert_equal(unstacked, mock_data_array)

    # Test that isolated NaNs raise an error
    with pytest.raises(ValueError):
        stacker.fit_transform(mock_data_array_isolated_nans, dim_sample, dim_feature)

    # Test that NaNs across a full dimension are handled correctly
    stacked = stacker.fit_transform(
        mock_data_array_full_dimensional_nans, dim_sample, dim_feature
    )
    unstacked = stacker.inverse_transform_data(stacked)
    xr.testing.assert_equal(unstacked, mock_data_array_full_dimensional_nans)

    # Test that NaNs on the boundary are handled correctly
    stacked = stacker.fit_transform(
        mock_data_array_boundary_nans, dim_sample, dim_feature
    )
    unstacked = stacker.inverse_transform_data(stacked)
    xr.testing.assert_equal(unstacked, mock_data_array_boundary_nans)

    # Test that the same stacker cannot be used with data of different shapes
    with pytest.raises(ValueError):
        other_data = mock_data_array.isel(time=slice(None, -1), lon=slice(None, -1))
        stacker.transform(other_data)


@pytest.mark.parametrize(
    "dim_sample, dim_feature",
    [
        (("time",), ("lat", "lon")),
        (("time",), ("lon", "lat")),
        (("lat", "lon"), ("time",)),
        (("lon", "lat"), ("time",)),
    ],
)
def test_transform(mock_data_array, dim_sample, dim_feature):
    # Test basic functionality
    stacker = SingleDataArrayStacker()
    stacker.fit_transform(mock_data_array, dim_sample, dim_feature)
    other_data = mock_data_array.copy(deep=True)
    transformed = stacker.transform(other_data)

    # Test that transformed data has the correct dimensions
    assert transformed.ndim == 2
    assert set(transformed.dims) == {"sample", "feature"}
    assert not transformed.isnull().any()

    # Invalid data raises an error
    with pytest.raises(ValueError):
        stacker.transform(mock_data_array.isel(lon=slice(None, 2), time=slice(None, 2)))


@pytest.mark.parametrize(
    "dim_sample, dim_feature",
    [
        (("time",), ("lat", "lon")),
        (("time",), ("lon", "lat")),
        (("lat", "lon"), ("time",)),
        (("lon", "lat"), ("time",)),
    ],
)
def test_inverse_transform_data(mock_data_array, dim_sample, dim_feature):
    # Test inverse transform
    stacker = SingleDataArrayStacker()
    stacker.fit_transform(mock_data_array, dim_sample, dim_feature)
    stacked = stacker.transform(mock_data_array)
    unstacked = stacker.inverse_transform_data(stacked)
    xr.testing.assert_equal(unstacked, mock_data_array)

    # Test that the operation is reversible
    restacked = stacker.transform(unstacked)
    xr.testing.assert_equal(restacked, stacked)


@pytest.mark.parametrize(
    "dim_sample, dim_feature",
    [
        (("time",), ("lat", "lon")),
        (("time",), ("lon", "lat")),
        (("lat", "lon"), ("time",)),
        (("lon", "lat"), ("time",)),
    ],
)
def test_inverse_transform_components(mock_data_array, dim_sample, dim_feature):
    # Test basic functionality
    stacker = SingleDataArrayStacker()
    stacker.fit_transform(mock_data_array, dim_sample, dim_feature)
    components = xr.DataArray(
        np.random.normal(size=(len(stacker.coords_out_["feature"]), 10)),
        dims=("feature", "mode"),
        coords={"feature": stacker.coords_out_["feature"]},
    )
    unstacked = stacker.inverse_transform_components(components)

    # Test that feature dimensions are preserved
    assert set(unstacked.dims) == set(dim_feature + ("mode",))

    # Test that feature coordinates are preserved
    for dim, coords in mock_data_array.coords.items():
        if dim in dim_feature:
            assert (
                unstacked.coords[dim].size == coords.size
            ), "Dimension {} has different size.".format(dim)


@pytest.mark.parametrize(
    "dim_sample, dim_feature",
    [
        (("time",), ("lat", "lon")),
        (("time",), ("lon", "lat")),
        (("lat", "lon"), ("time",)),
        (("lon", "lat"), ("time",)),
    ],
)
def test_inverse_transform_scores(mock_data_array, dim_sample, dim_feature):
    # Test basic functionality
    stacker = SingleDataArrayStacker()
    stacker.fit_transform(mock_data_array, dim_sample, dim_feature)
    scores = xr.DataArray(
        np.random.rand(len(stacker.coords_out_["sample"]), 10),
        dims=("sample", "mode"),
        coords={"sample": stacker.coords_out_["sample"]},
    )
    unstacked = stacker.inverse_transform_scores(scores)

    # Test that sample dimensions are preserved
    assert set(unstacked.dims) == set(dim_sample + ("mode",))

    # Test that sample coordinates are preserved
    for dim, coords in mock_data_array.coords.items():
        if dim in dim_sample:
            assert (
                unstacked.coords[dim].size == coords.size
            ), "Dimension {} has different size.".format(dim)
