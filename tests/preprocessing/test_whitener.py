import math

import pytest
import xarray as xr

from xeofs.preprocessing import Whitener

from ..conftest import generate_synthetic_dataarray
from ..utilities import (
    assert_expected_coords,
    assert_expected_dims,
    data_is_dask,
)

# =============================================================================
# GENERALLY VALID TEST CASES
# =============================================================================
N_SAMPLE_DIMS = [1]
N_FEATURE_DIMS = [1]
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
    "synthetic_dataarray",
    VALID_TEST_DATA,
    indirect=["synthetic_dataarray"],
)
def test_fit(synthetic_dataarray):
    data = synthetic_dataarray.rename({"sample0": "sample", "feature0": "feature"})

    whitener = Whitener(n_modes=2)
    whitener.fit(data)


@pytest.mark.parametrize(
    "synthetic_dataarray",
    VALID_TEST_DATA,
    indirect=["synthetic_dataarray"],
)
def test_transform(synthetic_dataarray):
    data = synthetic_dataarray.rename({"sample0": "sample", "feature0": "feature"})

    whitener = Whitener(n_modes=2)
    whitener.fit(data)

    # Transform data
    transformed_data = whitener.transform(data)
    transformed_data2 = whitener.transform(data)
    assert transformed_data.identical(transformed_data2)

    assert isinstance(transformed_data, xr.DataArray)
    assert transformed_data.ndim == 2
    assert transformed_data.dims == ("sample", "feature")

    # Consistent dask behaviour
    is_dask_before = data_is_dask(data)
    is_dask_after = data_is_dask(transformed_data)
    assert is_dask_before == is_dask_after


@pytest.mark.parametrize(
    "synthetic_dataarray",
    VALID_TEST_DATA,
    indirect=["synthetic_dataarray"],
)
def test_fit_transform(synthetic_dataarray):
    data = synthetic_dataarray.rename({"sample0": "sample", "feature0": "feature"})

    whitener = Whitener(n_modes=2)

    # Transform data
    transformed_data = whitener.fit_transform(data)
    transformed_data2 = whitener.transform(data)
    assert transformed_data.identical(transformed_data2)

    assert isinstance(transformed_data, xr.DataArray)
    assert transformed_data.ndim == 2
    assert transformed_data.dims == ("sample", "feature")

    # Consistent dask behaviour
    is_dask_before = data_is_dask(data)
    is_dask_after = data_is_dask(transformed_data)
    assert is_dask_before == is_dask_after


@pytest.mark.parametrize(
    "synthetic_dataarray",
    VALID_TEST_DATA,
    indirect=["synthetic_dataarray"],
)
def test_invserse_transform_data(synthetic_dataarray):
    data = synthetic_dataarray.rename({"sample0": "sample", "feature0": "feature"})

    whitener = Whitener(n_modes=2)
    whitener.fit(data)

    whitened_data = whitener.transform(data)
    unwhitened_data = whitener.inverse_transform_data(whitened_data)

    is_dask_before = data_is_dask(data)
    is_dask_after = data_is_dask(unwhitened_data)

    # Unstacked data has dimensions of original data
    assert_expected_dims(data, unwhitened_data, policy="all")
    # Unstacked data has coordinates of original data
    assert_expected_coords(data, unwhitened_data, policy="all")
    # inverse transform should not change dask-ness
    assert is_dask_before == is_dask_after


@pytest.mark.parametrize(
    "alpha",
    [0.0, 0.5, 1.0, 1.5],
)
def test_transform_alpha(alpha):
    data = generate_synthetic_dataarray(1, 1, "index", "no_nan", "no_dask")
    data = data.rename({"sample0": "sample", "feature0": "feature"})

    whitener = Whitener(n_modes=2, alpha=alpha)
    data_whitened = whitener.fit_transform(data)

    norm = (data_whitened**2).sum("sample")
    ones = norm / norm
    # Check that for alpha=0 full whitening is performed
    if math.isclose(alpha, 0.0, abs_tol=1e-6):
        xr.testing.assert_allclose(norm, ones, atol=1e-6)


@pytest.mark.parametrize(
    "alpha",
    [0.0, 0.5, 1.0, 1.5],
)
def test_invserse_transform_alpha(alpha):
    data = generate_synthetic_dataarray(1, 1, "index", "no_nan", "no_dask")
    data = data.rename({"sample0": "sample", "feature0": "feature"})

    whitener = Whitener(n_modes=6, alpha=alpha)
    data_whitened = whitener.fit_transform(data)
    data_unwhitened = whitener.inverse_transform_data(data_whitened)

    xr.testing.assert_allclose(data, data_unwhitened, atol=1e-6)


def test_invalid_alpha():
    data = generate_synthetic_dataarray(1, 1, "index", "no_nan", "no_dask")
    data = data.rename({"sample0": "sample", "feature0": "feature"})

    err_msg = "`alpha` must be greater than or equal to 0"
    with pytest.raises(ValueError, match=err_msg):
        Whitener(n_modes=2, alpha=-1.0)
