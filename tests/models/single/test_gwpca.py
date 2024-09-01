import pytest

import xeofs as xe

# =============================================================================
# GENERALLY VALID TEST CASES
# =============================================================================
N_ARRAYS = [1, 2]
N_SAMPLE_DIMS = [1, 2]
N_FEATURE_DIMS = [1, 2]
INDEX_POLICY = ["index"]
NAN_POLICY = ["no_nan", "fulldim"]
DASK_POLICY = ["no_dask"]
SEED = [0]

VALID_TEST_DATA = [
    (na, ns, nf, index, nan, dask)
    for na in N_ARRAYS
    for ns in N_SAMPLE_DIMS
    for nf in N_FEATURE_DIMS
    for index in INDEX_POLICY
    for nan in NAN_POLICY
    for dask in DASK_POLICY
]


# TESTS
# =============================================================================
@pytest.mark.parametrize(
    "kernel",
    [("bisquare"), ("gaussian"), ("exponential")],
)
def test_fit(mock_data_array, kernel):
    gwpca = xe.single.GWPCA(
        n_modes=2, metric="haversine", kernel=kernel, bandwidth=5000
    )
    gwpca.fit(mock_data_array, dim=("lat", "lon"))
    gwpca.components()
    gwpca.largest_locally_weighted_components()


@pytest.mark.parametrize(
    "metric, kernel, bandwidth",
    [
        ("haversine", "invalid_kernel", 5000),
        ("invalid_metric", "gaussian", 5000),
        ("haversine", "exponential", 0),
    ],
)
def test_fit_invalid(mock_data_array, metric, kernel, bandwidth):
    with pytest.raises(ValueError):
        xe.single.GWPCA(n_modes=2, metric=metric, kernel=kernel, bandwidth=bandwidth)
