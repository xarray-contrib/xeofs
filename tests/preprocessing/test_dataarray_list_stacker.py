import pytest
import xarray as xr
import numpy as np

from xeofs.preprocessing.stacker import ListDataArrayStacker


@pytest.mark.parametrize(
    "dim_sample, dim_feature",
    [
        (("time",), ("lat", "lon")),
        (("time",), ("lon", "lat")),
        (("lat", "lon"), ("time",)),
        (("lon", "lat"), ("time",)),
    ],
)
def test_data_array_list_stacker_fit_transform(
    dim_sample, dim_feature, mock_data_array_list
):
    """
    Test that ListDataArrayStacker correctly stacks a list of DataArrays and
    fit_transform returns DataArray with 'sample' and 'feature' dimensions.
    """
    stacker = ListDataArrayStacker()
    feature_dims_list = [dim_feature] * len(
        mock_data_array_list
    )  # Assume that all DataArrays have the same feature dimensions
    stacked_data = stacker.fit_transform(
        mock_data_array_list, dim_sample, feature_dims_list
    )

    # Check if the output is a DataArray
    assert isinstance(stacked_data, xr.DataArray)
    # Check if the dimensions are correct
    assert set(stacked_data.dims) == set(("sample", "feature"))
    # Check if the data is preserved
    assert stacked_data.size == sum([da.size for da in mock_data_array_list])

    # Check if the transform function returns the same result
    transformed_data = stacker.transform(mock_data_array_list)
    [
        xr.testing.assert_equal(stacked, transformed)
        for stacked, transformed in zip(stacked_data, transformed_data)
    ]

    # Check if the stacker dimensions are correct
    for stckr, da in zip(stacker.stackers, mock_data_array_list):
        assert set(stckr.dims_in_) == set(da.dims)
        assert set(stckr.dims_out_) == set(("sample", "feature"))
        # test that coordinates are preserved
        for dim, coords in da.coords.items():
            assert (
                stckr.coords_in_[dim].size == coords.size
            ), "Dimension {} has different size.".format(dim)
        assert stckr.coords_out_["sample"].size == np.prod(
            [coords.size for dim, coords in da.coords.items() if dim in dim_sample]
        ), "Sample dimension has different size."
        assert stckr.coords_out_["feature"].size == np.prod(
            [coords.size for dim, coords in da.coords.items() if dim in dim_feature]
        ), "Feature dimension has different size."

    # Check that invalid input raises an error in transform
    with pytest.raises(ValueError):
        stacker.transform(
            [
                xr.DataArray(np.random.rand(2, 4, 5), dims=("a", "y", "x"))
                for _ in range(3)
            ]
        )


@pytest.mark.parametrize(
    "dim_sample, dim_feature",
    [
        (("time",), ("lat", "lon")),
        (("time",), ("lon", "lat")),
        (("lat", "lon"), ("time",)),
        (("lon", "lat"), ("time",)),
    ],
)
def test_data_array_list_stacker_unstack_data(
    dim_sample, dim_feature, mock_data_array_list
):
    """Test if the inverse transformed DataArrays are identical to the original DataArrays."""
    stacker_list = ListDataArrayStacker()
    feature_dims_list = [dim_feature] * len(
        mock_data_array_list
    )  # Assume that all DataArrays have the same feature dimensions
    stacked = stacker_list.fit_transform(mock_data_array_list, dim_sample, feature_dims_list)  # type: ignore
    unstacked = stacker_list.inverse_transform_data(stacked)

    for da_test, da_ref in zip(unstacked, mock_data_array_list):
        xr.testing.assert_equal(da_test, da_ref)


@pytest.mark.parametrize(
    "dim_sample, dim_feature",
    [
        (("time",), ("lat", "lon")),
        (("time",), ("lon", "lat")),
        (("lat", "lon"), ("time",)),
        (("lon", "lat"), ("time",)),
    ],
)
def test_data_array_list_stacker_unstack_components(
    dim_sample, dim_feature, mock_data_array_list
):
    """Test if the inverse transformed components are identical to the original components."""
    stacker_list = ListDataArrayStacker()
    feature_dims_list = [dim_feature] * len(mock_data_array_list)
    stacked = stacker_list.fit_transform(
        mock_data_array_list, dim_sample, feature_dims_list
    )

    components = xr.DataArray(
        np.random.normal(size=(stacker_list.coords_out_["feature"].size, 10)),
        dims=("feature", "mode"),
        coords={"feature": stacker_list.coords_out_["feature"]},
    )
    unstacked = stacker_list.inverse_transform_components(components)

    for da_test, da_ref in zip(unstacked, mock_data_array_list):
        # Check if the dimensions are correct
        assert set(da_test.dims) == set(dim_feature + ("mode",))
        # Check if the coordinates are preserved
        for dim, coords in da_ref.coords.items():
            if dim in dim_feature:
                assert (
                    da_test.coords[dim].size == coords.size
                ), "Dimension {} has different size.".format(dim)
