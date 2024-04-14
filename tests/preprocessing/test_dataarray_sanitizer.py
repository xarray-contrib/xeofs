import pytest
import numpy as np

from xeofs.preprocessing.sanitizer import Sanitizer
from xeofs.utils.data_types import DataArray
from ..conftest import generate_synthetic_dataarray
from ..utilities import (
    data_is_dask,
    assert_expected_dims,
    assert_expected_coords,
)

# =============================================================================
# VALID TEST CASES
# =============================================================================
N_SAMPLE_DIMS = [1]
N_FEATURE_DIMS = [1]
INDEX_POLICY = ["index"]
NAN_POLICY = ["no_nan", "fulldim"]
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

FULL_NAN = [
    (ns, nf, index, nan, dask)
    for ns in N_SAMPLE_DIMS
    for nf in N_FEATURE_DIMS
    for index in INDEX_POLICY
    for nan in ["fulldim"]
    for dask in DASK_POLICY
]

ISOLATED_NAN = [
    (ns, nf, index, nan, dask)
    for ns in N_SAMPLE_DIMS
    for nf in N_FEATURE_DIMS
    for index in INDEX_POLICY
    for nan in ["isolated"]
    for dask in DASK_POLICY
]


# TESTS
# =============================================================================
@pytest.mark.parametrize(
    "sample_name, feature_name, data_params",
    [
        ("sample", "feature", (1, 1)),
        ("another_sample", "another_feature", (1, 1)),
    ],
)
def test_fit_valid_dimension_names(sample_name, feature_name, data_params):
    data = generate_synthetic_dataarray(*data_params)
    data = data.rename({"sample0": sample_name, "feature0": feature_name})

    sanitizer = Sanitizer(sample_name=sample_name, feature_name=feature_name)
    sanitizer.fit(data)
    data_clean = sanitizer.transform(data)
    reconstructed_data = sanitizer.inverse_transform_data(data_clean)

    assert data_clean.ndim == 2
    assert set(data_clean.dims) == set((sample_name, feature_name))
    assert set(reconstructed_data.dims) == set(data.dims)


@pytest.mark.parametrize(
    "sample_name, feature_name, data_params",
    [
        ("sample1", "feature", (1, 1)),
        ("sample", "feature1", (1, 1)),
        ("sample1", "feature1", (1, 1)),
    ],
)
def test_fit_invalid_dimension_names(sample_name, feature_name, data_params):
    data = generate_synthetic_dataarray(*data_params)

    sanitizer = Sanitizer(sample_name=sample_name, feature_name=feature_name)

    with pytest.raises(ValueError):
        sanitizer.fit(data)


@pytest.mark.parametrize(
    "synthetic_dataarray",
    VALID_TEST_DATA,
    indirect=["synthetic_dataarray"],
)
def test_transform(synthetic_dataarray):
    data = synthetic_dataarray
    data = data.rename({"sample0": "sample", "feature0": "feature"})

    sanitizer = Sanitizer()
    sanitizer.fit(data)
    transformed_data = sanitizer.transform(data)
    transformed_data2 = sanitizer.transform(data)

    is_dask_before = data_is_dask(data)
    is_dask_after = data_is_dask(transformed_data)

    assert transformed_data.notnull().all()
    assert isinstance(transformed_data, DataArray)
    assert transformed_data.ndim == 2
    assert transformed_data.dims == data.dims
    assert is_dask_before == is_dask_after
    assert transformed_data.identical(transformed_data2)


@pytest.mark.parametrize(
    "synthetic_dataarray",
    VALID_TEST_DATA,
    indirect=["synthetic_dataarray"],
)
def test_transform_invalid(synthetic_dataarray):
    data = synthetic_dataarray
    data = data.rename({"sample0": "sample", "feature0": "feature"})

    sanitizer = Sanitizer()
    sanitizer.fit(data)
    with pytest.raises(ValueError):
        sanitizer.transform(data.isel(feature0=slice(0, 2)))


@pytest.mark.parametrize(
    "synthetic_dataarray",
    VALID_TEST_DATA,
    indirect=["synthetic_dataarray"],
)
def test_fit_transform(synthetic_dataarray):
    data = synthetic_dataarray
    data = data.rename({"sample0": "sample", "feature0": "feature"})

    sanitizer = Sanitizer()
    transformed_data = sanitizer.fit_transform(data)

    is_dask_before = data_is_dask(data)
    is_dask_after = data_is_dask(transformed_data)

    assert isinstance(transformed_data, DataArray)
    assert transformed_data.notnull().all()
    assert transformed_data.ndim == 2
    assert transformed_data.dims == data.dims
    assert is_dask_before == is_dask_after


@pytest.mark.parametrize(
    "synthetic_dataarray",
    VALID_TEST_DATA,
    indirect=["synthetic_dataarray"],
)
def test_invserse_transform_data(synthetic_dataarray):
    data = synthetic_dataarray
    data = data.rename({"sample0": "sample", "feature0": "feature"})

    sanitizer = Sanitizer()
    sanitizer.fit(data)
    cleaned_data = sanitizer.transform(data)
    uncleaned_data = sanitizer.inverse_transform_data(cleaned_data)

    is_dask_before = data_is_dask(data)
    is_dask_after = data_is_dask(uncleaned_data)

    # inverse transform is only identical if nan_policy={"no_nan", "fulldim"}
    # in case of "isolated" the inverse transform will set the entire feature column
    # to NaNs, which is not identical to the original data
    # assert data.identical(uncleaned_data)

    # inverse transform should not change dask-ness
    assert is_dask_before == is_dask_after


@pytest.mark.parametrize(
    "synthetic_dataarray",
    VALID_TEST_DATA,
    indirect=["synthetic_dataarray"],
)
def test_invserse_transform_components(synthetic_dataarray):
    data: DataArray = synthetic_dataarray
    data = data.rename({"sample0": "sample", "feature0": "feature"})

    sanitizer = Sanitizer()
    sanitizer.fit(data)

    stacked_data = sanitizer.transform(data)
    components = stacked_data.rename({"sample": "mode"})
    unstacked_data = sanitizer.inverse_transform_components(components)

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
    data = data.rename({"sample0": "sample", "feature0": "feature"})

    sanitizer = Sanitizer()
    sanitizer.fit(data)

    stacked_data = sanitizer.transform(data)
    components = stacked_data.rename({"feature": "mode"})
    unstacked_data = sanitizer.inverse_transform_scores(components)

    is_dask_before = data_is_dask(data)
    is_dask_after = data_is_dask(unstacked_data)

    # Unstacked components has correct feature dimensions
    assert_expected_dims(data, unstacked_data, policy="sample")
    # Unstacked data has coordinates of original data
    assert_expected_coords(data, unstacked_data, policy="sample")
    # inverse transform should not change dask-ness
    assert is_dask_before == is_dask_after


@pytest.mark.parametrize(
    "synthetic_dataarray",
    ISOLATED_NAN,
    indirect=["synthetic_dataarray"],
)
def test_isolated_nans(synthetic_dataarray):
    data = synthetic_dataarray
    data = data.rename({"sample0": "sample", "feature0": "feature"})

    sanitizer = Sanitizer()
    sanitizer.fit(data)

    # By default want the sanitizer to raise for isolated NaNs
    with pytest.raises(ValueError):
        sanitizer.transform(data)

    # But allow this check to be disabled
    sanitizer.check_nans = False
    sanitizer.transform(data)


@pytest.mark.parametrize(
    "synthetic_dataarray",
    FULL_NAN,
    indirect=["synthetic_dataarray"],
)
def test_feature_nan_transform(synthetic_dataarray):
    data = synthetic_dataarray
    data = data.rename({"sample0": "sample", "feature0": "feature"})

    sanitizer = Sanitizer()
    sanitizer.fit(data)
    sanitizer.transform(data)

    # Pass through new data with NaNs in a different location
    data.loc[{"feature": 1}] = np.nan
    # By default want the sanitizer to raise on transform for new NaN features
    with pytest.raises(ValueError):
        sanitizer.transform(data)

    # But allow this check to be disabled
    sanitizer.check_nans = False
    sanitizer.transform(data)
