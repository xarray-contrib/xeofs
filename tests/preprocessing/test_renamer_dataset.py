import pytest

from xeofs.preprocessing.dimension_renamer import DimensionRenamer
from ..utilities import (
    data_is_dask,
    get_dims_from_data,
)

# =============================================================================
# GENERALLY VALID TEST CASES
# =============================================================================
N_VARIABLES = [1, 2]
N_SAMPLE_DIMS = [1, 2]
N_FEATURE_DIMS = [1, 2]
INDEX_POLICY = ["index", "multiindex"]
NAN_POLICY = ["no_nan", "fulldim"]
DASK_POLICY = ["no_dask", "dask"]
SEED = [0]

VALID_TEST_DATA = [
    (nv, ns, nf, index, nan, dask)
    for nv in N_VARIABLES
    for ns in N_SAMPLE_DIMS
    for nf in N_FEATURE_DIMS
    for index in INDEX_POLICY
    for nan in NAN_POLICY
    for dask in DASK_POLICY
]


# TESTS
# =============================================================================
@pytest.mark.parametrize(
    "synthetic_dataset",
    VALID_TEST_DATA,
    indirect=["synthetic_dataset"],
)
def test_transform(synthetic_dataset):
    all_dims, sample_dims, feature_dims = get_dims_from_data(synthetic_dataset)

    n_dims = len(all_dims)

    base = "new"
    start = 10
    expected_dims = set(base + str(i) for i in range(start, start + n_dims))

    renamer = DimensionRenamer(base=base, start=start)
    renamer.fit(synthetic_dataset, sample_dims, feature_dims)
    transformed_data = renamer.transform(synthetic_dataset)

    is_dask_before = data_is_dask(synthetic_dataset)
    is_dask_after = data_is_dask(transformed_data)

    # Transforming doesn't change the dask-ness of the data
    assert is_dask_before == is_dask_after

    # Transforming converts dimension names
    given_dims = set(transformed_data.dims)
    assert given_dims == expected_dims

    # Result is robust to calling the method multiple times
    transformed_data = renamer.transform(synthetic_dataset)
    given_dims = set(transformed_data.dims)
    assert given_dims == expected_dims


@pytest.mark.parametrize(
    "synthetic_dataset",
    VALID_TEST_DATA,
    indirect=["synthetic_dataset"],
)
def test_inverse_transform_data(synthetic_dataset):
    all_dims, sample_dims, feature_dims = get_dims_from_data(synthetic_dataset)

    base = "new"
    start = 10

    renamer = DimensionRenamer(base=base, start=start)
    renamer.fit(synthetic_dataset, sample_dims, feature_dims)
    transformed_data = renamer.transform(synthetic_dataset)
    inverse_transformed_data = renamer.inverse_transform_data(transformed_data)

    is_dask_before = data_is_dask(synthetic_dataset)
    is_dask_after = data_is_dask(transformed_data)

    # Transforming doesn't change the dask-ness of the data
    assert is_dask_before == is_dask_after

    assert inverse_transformed_data.identical(synthetic_dataset)
    assert set(inverse_transformed_data.dims) == set(synthetic_dataset.dims)
