import pytest

from xeofs.preprocessing.dimension_renamer import DimensionRenamer
from ..utilities import (
    data_is_dask,
    get_dims_from_data,
)

# =============================================================================
# GENERALLY VALID TEST CASES
# =============================================================================
N_SAMPLE_DIMS = [1, 2]
N_FEATURE_DIMS = [1, 2]
INDEX_POLICY = ["index", "multiindex"]
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


@pytest.mark.parametrize(
    "synthetic_dataarray",
    VALID_TEST_DATA,
    indirect=["synthetic_dataarray"],
)
def test_transform(synthetic_dataarray):
    all_dims, sample_dims, feature_dims = get_dims_from_data(synthetic_dataarray)

    n_dims = len(all_dims)

    base = "new"
    start = 10
    expected_dims = set(base + str(i) for i in range(start, start + n_dims))

    renamer = DimensionRenamer(base=base, start=start)
    renamer.fit(synthetic_dataarray, sample_dims, feature_dims)
    transformed_data = renamer.transform(synthetic_dataarray)

    is_dask_before = data_is_dask(synthetic_dataarray)
    is_dask_after = data_is_dask(transformed_data)

    # Transforming doesn't change the dask-ness of the data
    assert is_dask_before == is_dask_after

    # Transforming converts dimension names
    given_dims = set(transformed_data.dims)
    assert given_dims == expected_dims

    # Result is robust to calling the method multiple times
    transformed_data = renamer.transform(synthetic_dataarray)
    given_dims = set(transformed_data.dims)
    assert given_dims == expected_dims


@pytest.mark.parametrize(
    "synthetic_dataarray",
    VALID_TEST_DATA,
    indirect=["synthetic_dataarray"],
)
def test_inverse_transform_data(synthetic_dataarray):
    all_dims, sample_dims, feature_dims = get_dims_from_data(synthetic_dataarray)

    base = "new"
    start = 10

    renamer = DimensionRenamer(base=base, start=start)
    renamer.fit(synthetic_dataarray, sample_dims, feature_dims)
    transformed_data = renamer.transform(synthetic_dataarray)
    inverse_transformed_data = renamer.inverse_transform_data(transformed_data)

    is_dask_before = data_is_dask(synthetic_dataarray)
    is_dask_after = data_is_dask(transformed_data)

    # Transforming doesn't change the dask-ness of the data
    assert is_dask_before == is_dask_after

    assert inverse_transformed_data.identical(synthetic_dataarray)
    assert set(inverse_transformed_data.dims) == set(synthetic_dataarray.dims)
