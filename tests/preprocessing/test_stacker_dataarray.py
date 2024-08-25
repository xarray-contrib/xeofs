import pytest

from xeofs.preprocessing import Stacker
from xeofs.utils.data_types import DataArray
from ..conftest import generate_synthetic_dataarray
from ..utilities import (
    get_dims_from_data,
    data_is_dask,
    assert_expected_dims,
    assert_expected_coords,
)

# =============================================================================
# GENERALLY VALID TEST CASES
# =============================================================================
N_SAMPLE_DIMS = [1, 2]
N_FEATURE_DIMS = [1, 2]
INDEX_POLICY = ["index"]
NAN_POLICY = ["no_nan"]
DASK_POLICY = ["no_dask", "dask"]
SEED = [0]

VALID_TEST_DATA = [
    (ns, nf, index, nan, dask)
    for ns in N_SAMPLE_DIMS
    for nf in N_FEATURE_DIMS
    for index in INDEX_POLICY
    for nan in NAN_POLICY
    for dask in DASK_POLICY
]


# TESTS
# =============================================================================
@pytest.mark.parametrize(
    "sample_name, feature_name, data_params",
    [
        ("sample", "feature", (1, 1)),
        ("sample0", "feature0", (1, 1)),
        ("sample0", "feature", (1, 2)),
        ("sample", "feature0", (2, 1)),
        ("sample", "feature", (2, 2)),
        ("another_sample", "another_feature", (1, 1)),
        ("another_sample", "another_feature", (2, 2)),
    ],
)
def test_fit_valid_dimension_names(sample_name, feature_name, data_params):
    data = generate_synthetic_dataarray(*data_params)
    all_dims, sample_dims, feature_dims = get_dims_from_data(data)

    stacker = Stacker(sample_name=sample_name, feature_name=feature_name)
    stacker.fit(data, sample_dims, feature_dims)
    stacked_data = stacker.transform(data)
    reconstructed_data = stacker.inverse_transform_data(stacked_data)

    assert stacked_data.ndim == 2
    assert set(stacked_data.dims) == set((sample_name, feature_name))
    assert set(reconstructed_data.dims) == set(data.dims)


@pytest.mark.parametrize(
    "sample_name, feature_name, data_params",
    [
        ("sample1", "feature", (2, 1)),
        ("sample", "feature1", (1, 2)),
        ("sample1", "feature1", (3, 3)),
    ],
)
def test_fit_invalid_dimension_names(sample_name, feature_name, data_params):
    data = generate_synthetic_dataarray(*data_params)
    all_dims, sample_dims, feature_dims = get_dims_from_data(data)

    stacker = Stacker(sample_name=sample_name, feature_name=feature_name)

    with pytest.raises(ValueError):
        stacker.fit(data, sample_dims, feature_dims)


@pytest.mark.parametrize(
    "synthetic_dataarray",
    VALID_TEST_DATA,
    indirect=["synthetic_dataarray"],
)
def test_fit(synthetic_dataarray):
    data = synthetic_dataarray
    all_dims, sample_dims, feature_dims = get_dims_from_data(data)

    stacker = Stacker()
    stacker.fit(data, sample_dims, feature_dims)


@pytest.mark.parametrize(
    "synthetic_dataarray",
    VALID_TEST_DATA,
    indirect=["synthetic_dataarray"],
)
def test_transform(synthetic_dataarray):
    data = synthetic_dataarray
    all_dims, sample_dims, feature_dims = get_dims_from_data(data)

    stacker = Stacker()
    stacker.fit(data, sample_dims, feature_dims)
    transformed_data = stacker.transform(data)
    transformed_data2 = stacker.transform(data)

    is_dask_before = data_is_dask(data)
    is_dask_after = data_is_dask(transformed_data)

    assert isinstance(transformed_data, DataArray)
    assert transformed_data.ndim == 2
    assert transformed_data.dims == ("sample", "feature")
    assert is_dask_before == is_dask_after
    assert transformed_data.identical(transformed_data2)


@pytest.mark.parametrize(
    "synthetic_dataarray",
    VALID_TEST_DATA,
    indirect=["synthetic_dataarray"],
)
def test_transform_invalid(synthetic_dataarray):
    data = synthetic_dataarray
    all_dims, sample_dims, feature_dims = get_dims_from_data(data)

    stacker = Stacker()
    stacker.fit(data, sample_dims, feature_dims)
    with pytest.raises(ValueError):
        stacker.transform(data.isel(feature0=slice(0, 2)))


@pytest.mark.parametrize(
    "synthetic_dataarray",
    VALID_TEST_DATA,
    indirect=["synthetic_dataarray"],
)
def test_fit_transform(synthetic_dataarray):
    data = synthetic_dataarray
    all_dims, sample_dims, feature_dims = get_dims_from_data(data)

    stacker = Stacker()
    transformed_data = stacker.fit_transform(data, sample_dims, feature_dims)

    is_dask_before = data_is_dask(data)
    is_dask_after = data_is_dask(transformed_data)

    assert isinstance(transformed_data, DataArray)
    assert transformed_data.ndim == 2
    assert transformed_data.dims == ("sample", "feature")
    assert is_dask_before == is_dask_after


@pytest.mark.parametrize(
    "synthetic_dataarray",
    VALID_TEST_DATA,
    indirect=["synthetic_dataarray"],
)
def test_invserse_transform_data(synthetic_dataarray):
    data = synthetic_dataarray
    all_dims, sample_dims, feature_dims = get_dims_from_data(data)

    stacker = Stacker()
    stacker.fit(data, sample_dims, feature_dims)
    stacked_data = stacker.transform(data)
    unstacked_data = stacker.inverse_transform_data(stacked_data)

    is_dask_before = data_is_dask(data)
    is_dask_after = data_is_dask(unstacked_data)

    # Unstacked data has dimensions of original data
    assert_expected_dims(data, unstacked_data, policy="all")
    # Unstacked data has coordinates of original data
    assert_expected_coords(data, unstacked_data, policy="all")
    # inverse transform should not change dask-ness
    assert is_dask_before == is_dask_after


@pytest.mark.parametrize(
    "synthetic_dataarray",
    VALID_TEST_DATA,
    indirect=["synthetic_dataarray"],
)
def test_invserse_transform_components(synthetic_dataarray):
    data: DataArray = synthetic_dataarray
    all_dims, sample_dims, feature_dims = get_dims_from_data(data)

    stacker = Stacker()
    stacker.fit(data, sample_dims, feature_dims)

    stacked_data = stacker.transform(data)
    components = stacked_data.rename({"sample": "mode"})
    unstacked_data = stacker.inverse_transform_components(components)

    is_dask_before = data_is_dask(data)
    is_dask_after = data_is_dask(unstacked_data)

    # Unstacked components has correct feature dimensions
    assert_expected_dims(data, unstacked_data, policy="feature")
    # Unstacked data has coordinates of original data
    assert_expected_coords(data, unstacked_data, policy="feature")
    # inverse transform should not change dask-ness
    assert is_dask_before == is_dask_after


@pytest.mark.parametrize(
    "synthetic_dataarray",
    VALID_TEST_DATA,
    indirect=["synthetic_dataarray"],
)
def test_invserse_transform_scores(synthetic_dataarray):
    data: DataArray = synthetic_dataarray
    all_dims, sample_dims, feature_dims = get_dims_from_data(data)

    stacker = Stacker()
    stacker.fit(data, sample_dims, feature_dims)

    stacked_data = stacker.transform(data)
    components = stacked_data.rename({"feature": "mode"})
    unstacked_data = stacker.inverse_transform_scores(components)

    is_dask_before = data_is_dask(data)
    is_dask_after = data_is_dask(unstacked_data)

    # Unstacked components has correct feature dimensions
    assert_expected_dims(data, unstacked_data, policy="sample")
    # Unstacked data has coordinates of original data
    assert_expected_coords(data, unstacked_data, policy="sample")
    # inverse transform should not change dask-ness
    assert is_dask_before == is_dask_after
