import numpy as np
import pytest
import xarray as xr

from xeofs.preprocessing import PCA

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
@pytest.mark.parametrize("lazy", [False, True])
def test_fit(lazy):
    data = generate_well_conditioned_data(lazy)

    pca = PCA(n_modes=2)
    pca.fit(data)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("use_pca", [True, False])
def test_transform(lazy, use_pca):
    data = generate_well_conditioned_data(lazy)

    pca = PCA(n_modes=2, use_pca=use_pca)
    pca.fit(data)

    # Transform data
    transformed_data = pca.transform(data)
    transformed_data2 = pca.transform(data)
    assert transformed_data.identical(transformed_data2)

    assert isinstance(transformed_data, xr.DataArray)
    assert transformed_data.ndim == 2
    assert transformed_data.dims == ("sample", "feature")

    # Consistent dask behaviour
    is_dask_before = data_is_dask(data)
    is_dask_after = data_is_dask(transformed_data)
    assert is_dask_before == is_dask_after


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("use_pca", [True, False])
def test_fit_transform(lazy, use_pca):
    data = generate_well_conditioned_data(lazy)

    pca = PCA(n_modes=2, use_pca=use_pca)

    # Transform data
    transformed_data = pca.fit_transform(data)
    transformed_data2 = pca.transform(data)
    assert transformed_data.identical(transformed_data2)

    assert isinstance(transformed_data, xr.DataArray)
    assert transformed_data.ndim == 2
    assert transformed_data.dims == ("sample", "feature")

    # Consistent dask behaviour
    is_dask_before = data_is_dask(data)
    is_dask_after = data_is_dask(transformed_data)
    assert is_dask_before == is_dask_after


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("use_pca", [True, False])
def test_invserse_transform_data(lazy, use_pca):
    data = generate_well_conditioned_data(lazy)

    pca = PCA(n_modes=2, use_pca=use_pca)
    pca.fit(data)

    transformed = pca.transform(data)
    untransformed = pca.inverse_transform_data(transformed)

    is_dask_before = data_is_dask(data)
    is_dask_after = data_is_dask(untransformed)

    # Unstacked data has dimensions of original data
    assert_expected_dims(data, untransformed, policy="all")
    # Unstacked data has coordinates of original data
    assert_expected_coords(data, untransformed, policy="all")
    # inverse transform should not change dask-ness
    assert is_dask_before == is_dask_after


@pytest.mark.parametrize("n_modes", [1, 2, 3])
def test_transform_pca_n_modes(n_modes):
    data = generate_well_conditioned_data()

    pca = PCA(use_pca=True, n_modes=n_modes)
    transformed = pca.fit_transform(data)

    # PCA reduces dimensionality
    assert transformed.shape[1] == n_modes


@pytest.mark.parametrize("use_pca", [True, False])
def test_transform_keep_coordinates(use_pca):
    X = generate_well_conditioned_data()

    pca = PCA(use_pca=use_pca, n_modes="all")
    transformed = pca.fit_transform(X)

    assert len(transformed.coords) == len(X.coords)
