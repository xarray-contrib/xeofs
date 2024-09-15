import math

import numpy as np
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


def generate_well_conditioned_data(lazy=False):
    t = np.linspace(0, 50, 200)
    std = 0.1
    X = np.sin(t)[:, None] + np.random.normal(0, std, size=(200, 3))
    X[:, 1] = X[:, 1] ** 3
    X[:, 2] = abs(X[:, 2]) ** (0.5)
    X = xr.DataArray(
        X,
        dims=["sample", "feature"],
        coords={"sample": np.arange(200), "feature": np.arange(3)},
        name="X",
    )
    X = X - X.mean("sample")
    if lazy:
        X = X.chunk({"sample": 5, "feature": -1})
    return X


# TESTS
# =============================================================================
@pytest.mark.parametrize(
    "lazy",
    [False, True],
)
def test_fit(lazy):
    data = generate_well_conditioned_data(lazy)

    whitener = Whitener()
    whitener.fit(data)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 1.5])
def test_transform(lazy, alpha):
    data = generate_well_conditioned_data(lazy)

    whitener = Whitener()
    whitener.fit(data)

    # Transform data
    transformed_data = whitener.transform(data)
    transformed_data2 = whitener.transform(data)

    assert isinstance(transformed_data, xr.DataArray)
    assert transformed_data.ndim == 2
    assert transformed_data.dims == ("sample", "feature")

    # Transformed data is identical
    assert transformed_data.identical(transformed_data2)

    # Check that for full whitening, transformed data is uncorrelated
    # and has variance one
    if math.isclose(alpha, 0.0, abs_tol=1e-6):
        # Transformed data has variance = 1
        assert np.allclose(transformed_data.var("sample").values, 1.0)

        # Transformed data is uncorrelated
        C = np.corrcoef(transformed_data.values, rowvar=False)
        target = np.identity(data.shape[1])
        np.testing.assert_allclose(C, target, atol=1e-6)

    # Consistent dask behaviour
    is_dask_before = data_is_dask(data)
    is_dask_after = data_is_dask(transformed_data)
    assert is_dask_before == is_dask_after


@pytest.mark.parametrize("lazy", [False, True])
def test_fit_transform(lazy):
    data = generate_well_conditioned_data(lazy)

    whitener = Whitener()

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


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 1.5])
def test_inverse_transform_data(lazy, alpha):
    data = generate_well_conditioned_data(lazy)

    whitener = Whitener(alpha)
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

    # Unwhitened data is identical to original data
    xr.testing.assert_allclose(data, unwhitened_data, atol=1e-6)


def test_invalid_alpha():
    data = generate_synthetic_dataarray(1, 1, "index", "no_nan", "no_dask")
    data = data.rename({"sample0": "sample", "feature0": "feature"})

    err_msg = "`alpha` must be greater than or equal to 0"
    with pytest.raises(ValueError, match=err_msg):
        Whitener(alpha=-1.0)


def test_raise_warning_ill_conditioned():
    # Synthetic dataset has 6 samples and 7 features
    data = generate_synthetic_dataarray(1, 1, "index", "no_nan", "no_dask")
    data = data.rename({"sample0": "sample", "feature0": "feature"})

    whitener = Whitener()
    warn_msg = "The number of samples (.*) is smaller than the number of features (.*), leading to an ill-conditioned problem.*"
    with pytest.warns(match=warn_msg):
        _ = whitener.fit_transform(data)


def test_whitener_identity_transformation():
    data = generate_well_conditioned_data()

    whitener = Whitener(alpha=1.0)
    transformed = whitener.fit_transform(data)
    reconstructed = whitener.inverse_transform_data(transformed)

    # No transformation takes place
    xr.testing.assert_identical(data, transformed)
    xr.testing.assert_identical(data, reconstructed)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 1.5])
def test_unwhiten_cross_covariance_matrix(alpha):
    def unwhiten_cov_mat(Cw, S1_inv, S2_inv):
        n_dims_s1 = len(S1_inv.shape)
        n_dims_s2 = len(S2_inv.shape)

        match n_dims_s1:
            case 0:
                C_unwhitened = Cw
            case 2:
                C_unwhitened = S1_inv @ Cw
            case _:
                raise ValueError("Invalid number of dimensions for S1_inv")

        match n_dims_s2:
            case 0:
                pass
            case 2:
                C_unwhitened = C_unwhitened @ S2_inv
            case _:
                raise ValueError("Invalid number of dimensions for S2_inv")

        return C_unwhitened

    """Test that we can uncover the original total amount of squared covariance between two datasets after whitening."""
    data1 = generate_well_conditioned_data()
    data2 = generate_well_conditioned_data() ** 2

    whitener1 = Whitener(alpha=0.5)
    whitener2 = Whitener(alpha=alpha)

    transformed1 = whitener1.fit_transform(data1)
    transformed2 = whitener2.fit_transform(data2)

    S1_inv = whitener1.Tinv.values
    S2_inv = whitener2.Tinv.values

    C = data1.values.T @ data2.values
    Cw = transformed1.values.T @ transformed2.values

    C_rec = unwhiten_cov_mat(Cw, S1_inv, S2_inv)

    total_covariance = np.linalg.norm(C)
    total_covariance_rec = np.linalg.norm(C_rec)
    np.testing.assert_almost_equal(total_covariance, total_covariance_rec)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 1.5])
def test_transform_keep_coordinates(alpha):
    X = generate_well_conditioned_data()

    whitener = Whitener(alpha=alpha)
    transformed = whitener.fit_transform(X)

    assert len(transformed.coords) == len(X.coords)
