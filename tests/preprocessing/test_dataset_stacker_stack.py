import pytest
import xarray as xr
import numpy as np

from xeofs.preprocessing.stacker import SingleDatasetStacker


def create_ds(dim_sample, dim_feature, seed=None):
    n_dims = len(dim_sample) + len(dim_feature)
    size = n_dims * [3]
    rng = np.random.default_rng(seed)
    dims = dim_sample + dim_feature
    coords = {d: np.arange(i, i + 3) for i, d in enumerate(dims)}
    da1 = xr.DataArray(rng.normal(0, 1, size=size), dims=dims, coords=coords)
    da2 = da1.sel(**{dim_feature[0]: slice(0, 2)}).squeeze().copy()
    ds = xr.Dataset({"da1": da1, "da2": da2})
    return ds


# Valid input
# =============================================================================
valid_input_dims = [
    # This SHOULD work but currently doesn't, potentially due to a bug in xarray (https://github.com/pydata/xarray/discussions/8063)
    # (("year", "month"), ("lon", "lat")),
    (("year",), ("lat", "lon")),
    (("year", "month"), ("lon",)),
    (("year",), ("lon",)),
]

valid_input = []
for dim_sample, dim_feature in valid_input_dims:
    da = create_ds(dim_sample, dim_feature)
    valid_input.append((da, dim_sample, dim_feature))


# Invalid input
# =============================================================================
invalid_input_dims = [
    (("sample",), ("feature", "lat")),
    (("sample",), ("month", "feature")),
    (("sample", "month"), ("lon", "lat")),
    (("sample",), ("lon", "lat")),
    (("year",), ("month", "sample")),
    (("year",), ("sample",)),
    (("sample",), ("lon",)),
    (("year", "month"), ("lon", "feature")),
    (("year", "month"), ("feature",)),
    (("year",), ("feature",)),
    (("feature",), ("lon", "lat")),
    (("feature",), ("lon",)),
    (("feature",), ("sample",)),
    (("sample",), ("feature",)),
]
invalid_input = []
for dim_sample, dim_feature in invalid_input_dims:
    da = create_ds(dim_sample, dim_feature)
    invalid_input.append((da, dim_sample, dim_feature))


# Test stacking
# =============================================================================
@pytest.mark.parametrize("da, dim_sample, dim_feature", valid_input)
def test_fit_transform(da, dim_sample, dim_feature):
    """Test fit_transform with valid input."""
    stacker = SingleDatasetStacker()
    da_stacked = stacker.fit_transform(da, dim_sample, dim_feature)

    # Stacked data has dimensions (sample, feature)
    err_msg = f"In: {da.dims}; Out: {da_stacked.dims}"
    assert set(da_stacked.dims) == {
        "sample",
        "feature",
    }, err_msg


@pytest.mark.parametrize("da, dim_sample, dim_feature", invalid_input)
def test_fit_transform_invalid_input(da, dim_sample, dim_feature):
    """Test fit_transform with invalid input."""
    stacker = SingleDatasetStacker()
    with pytest.raises(ValueError):
        da_stacked = stacker.fit_transform(da, dim_sample, dim_feature)


@pytest.mark.parametrize("da, dim_sample, dim_feature", valid_input)
def test_inverse_transform_data(da, dim_sample, dim_feature):
    """Test inverse transform with valid input."""
    stacker = SingleDatasetStacker()
    da_stacked = stacker.fit_transform(da, dim_sample, dim_feature)
    da_unstacked = stacker.inverse_transform_data(da_stacked)

    # Unstacked data has dimensions of original data
    err_msg = f"Original: {da.dims}; Recovered: {da_unstacked.dims}"
    assert set(da_unstacked.dims) == set(da.dims), err_msg
    # Unstacked data has variables of original data
    err_msg = f"Original: {set(da.data_vars)}; Recovered: {set(da_unstacked.data_vars)}"
    assert set(da_unstacked.data_vars) == set(da.data_vars), err_msg
    # Unstacked data has coordinates of original data
    for var in da.data_vars:
        # Check if the dimensions are correct
        err_msg = f"Original: {da[var].dims}; Recovered: {da_unstacked[var].dims}"
        assert set(da_unstacked[var].dims) == set(da[var].dims), err_msg
        for coord in da[var].coords:
            err_msg = f"Original: {da[var].coords[coord]}; Recovered: {da_unstacked[var].coords[coord]}"
            coords_are_equal = da[var].coords[coord] == da_unstacked[var].coords[coord]
            assert np.all(coords_are_equal), err_msg


@pytest.mark.parametrize("da, dim_sample, dim_feature", valid_input)
def test_inverse_transform_components(da, dim_sample, dim_feature):
    """Test inverse transform components with valid input."""
    stacker = SingleDatasetStacker()
    da_stacked = stacker.fit_transform(da, dim_sample, dim_feature)
    # Mock components by dropping sampling dim from data
    comps_stacked = da_stacked.drop_vars("sample").rename({"sample": "mode"})
    comps_stacked.coords.update({"mode": range(comps_stacked.mode.size)})

    comps_unstacked = stacker.inverse_transform_components(comps_stacked)

    # Unstacked components are a Dataset
    assert isinstance(comps_unstacked, xr.Dataset)

    # Unstacked data has variables of original data
    err_msg = (
        f"Original: {set(da.data_vars)}; Recovered: {set(comps_unstacked.data_vars)}"
    )
    assert set(comps_unstacked.data_vars) == set(da.data_vars), err_msg

    # Unstacked components has correct feature dimensions
    expected_dims = dim_feature + ("mode",)
    err_msg = f"Expected: {expected_dims}; Recovered: {comps_unstacked.dims}"
    assert set(comps_unstacked.dims) == set(expected_dims), err_msg

    # Unstacked components has coordinates of original data
    for var in da.data_vars:
        # Feature dimensions in original and recovered comps are same for each variable
        expected_dims_in_var = tuple(d for d in da[var].dims if d in dim_feature)
        expected_dims_in_var += ("mode",)
        err_msg = (
            f"Original: {expected_dims_in_var}; Recovered: {comps_unstacked[var].dims}"
        )
        assert set(expected_dims_in_var) == set(comps_unstacked[var].dims), err_msg
        # Coordinates in original and recovered comps are same for each variable
        for dim in expected_dims_in_var:
            if dim != "mode":
                err_msg = f"Original: {da[var].coords[dim]}; Recovered: {comps_unstacked[var].coords[dim]}"
                coords_are_equal = (
                    da[var].coords[dim] == comps_unstacked[var].coords[dim]
                )
                assert np.all(coords_are_equal), err_msg


@pytest.mark.parametrize("da, dim_sample, dim_feature", valid_input)
def test_inverse_transform_scores(da, dim_sample, dim_feature):
    """Test inverse transform scores with valid input."""
    stacker = SingleDatasetStacker()
    da_stacked = stacker.fit_transform(da, dim_sample, dim_feature)
    # Mock scores by dropping feature dim from data
    scores_stacked = da_stacked.drop_vars("feature").rename({"feature": "mode"})
    scores_stacked.coords.update({"mode": range(scores_stacked.mode.size)})

    scores_unstacked = stacker.inverse_transform_scores(scores_stacked)

    # Unstacked scores are a DataArray
    assert isinstance(scores_unstacked, xr.DataArray)
    # Unstacked components has correct feature dimensions
    expected_dims = dim_sample + ("mode",)
    err_msg = f"Expected: {expected_dims}; Recovered: {scores_unstacked.dims}"
    assert set(scores_unstacked.dims) == set(expected_dims), err_msg
    # Unstacked data has coordinates of original data
    for d in dim_sample:
        assert np.all(scores_unstacked.coords[d].values == da.coords[d].values)
