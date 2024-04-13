import pytest
import numpy as np

from xeofs.preprocessing.preprocessor import Preprocessor
from ..conftest import generate_synthetic_dataset
from ..utilities import (
    get_dims_from_data,
    data_is_dask,
    data_has_multiindex,
    assert_expected_dims,
    assert_expected_coords,
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

TEST_DATA_PARAMS = [
    (nv, ns, nf, index, nan, dask)
    for nv in N_VARIABLES
    for ns in N_SAMPLE_DIMS
    for nf in N_FEATURE_DIMS
    for index in INDEX_POLICY
    for nan in NAN_POLICY
    for dask in DASK_POLICY
]

SAMPLE_DIM_NAMES = ["sample"]
FEATURE_DIM_NAMES = ["feature", "feature_alternative"]

VALID_TEST_CASES = [
    (sample_name, feature_name, data_params)
    for sample_name in SAMPLE_DIM_NAMES
    for feature_name in FEATURE_DIM_NAMES
    for data_params in TEST_DATA_PARAMS
]


# TESTS
# =============================================================================
@pytest.mark.parametrize(
    "with_std, with_coslat, with_weights",
    [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, True, True),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ],
)
def test_fit_transform_scalings(with_std, with_coslat, with_weights, mock_dataset):
    """fit method should not be implemented."""
    prep = Preprocessor(with_std=with_std, with_coslat=with_coslat)

    weights = None
    if with_weights:
        weights = mock_dataset.mean("time").copy()
        weights = weights.where(weights is True, 0.5)

    data_trans = prep.fit_transform(mock_dataset, "time", weights)

    assert hasattr(prep, "scaler")
    assert hasattr(prep, "renamer")
    assert hasattr(prep, "preconverter")
    assert hasattr(prep, "stacker")
    assert hasattr(prep, "postconverter")
    assert hasattr(prep, "sanitizer")

    # Transformed data is centered
    assert np.isclose(data_trans.mean("sample"), 0).all()
    # Transformed data is standardized
    if with_std and not with_coslat:
        if with_weights:
            assert np.isclose(data_trans.std("sample"), 0.5).all()
        else:
            assert np.isclose(data_trans.std("sample"), 1).all()


@pytest.mark.parametrize(
    "index_policy, nan_policy, dask_policy",
    [
        ("index", "no_nan", "no_dask"),
        ("multiindex", "no_nan", "dask"),
        ("index", "fulldim", "no_dask"),
        ("multiindex", "fulldim", "dask"),
    ],
)
def test_fit_transform_same_dim_names(index_policy, nan_policy, dask_policy):
    data = generate_synthetic_dataset(1, 1, 1, index_policy, nan_policy, dask_policy)
    all_dims, sample_dims, feature_dims = get_dims_from_data(data)

    prep = Preprocessor(sample_name="sample0", feature_name="feature")
    transformed = prep.fit_transform(data, sample_dims)
    reconstructed = prep.inverse_transform_data(transformed)

    data_is_dask_before = data_is_dask(data)
    data_is_dask_interm = data_is_dask(transformed)
    data_is_dask_after = data_is_dask(reconstructed)

    assert set(transformed.dims) == set(("sample0", "feature"))
    assert set(reconstructed.dims) == set(("sample0", "feature0"))
    assert not data_has_multiindex(transformed)
    assert transformed.notnull().all()
    assert data_is_dask_before == data_is_dask_interm
    assert data_is_dask_before == data_is_dask_after


@pytest.mark.parametrize(
    "sample_name, feature_name, data_params",
    VALID_TEST_CASES,
)
def test_fit_transform(sample_name, feature_name, data_params):
    data = generate_synthetic_dataset(*data_params)
    all_dims, sample_dims, feature_dims = get_dims_from_data(data)

    prep = Preprocessor(sample_name=sample_name, feature_name=feature_name)
    transformed = prep.fit_transform(data, sample_dims)

    data_is_dask_before = data_is_dask(data)
    data_is_dask_after = data_is_dask(transformed)

    assert transformed.dims == (sample_name, feature_name)
    assert not data_has_multiindex(transformed)
    assert transformed.notnull().all()
    assert data_is_dask_before == data_is_dask_after


@pytest.mark.parametrize(
    "sample_name, feature_name, data_params",
    VALID_TEST_CASES,
)
def test_inverse_transform(sample_name, feature_name, data_params):
    data = generate_synthetic_dataset(*data_params)
    all_dims, sample_dims, feature_dims = get_dims_from_data(data)

    prep = Preprocessor(sample_name=sample_name, feature_name=feature_name)
    transformed = prep.fit_transform(data, sample_dims)
    components = transformed.rename({sample_name: "mode"})
    scores = transformed.rename({feature_name: "mode"})

    reconstructed = prep.inverse_transform_data(transformed)
    components = prep.inverse_transform_components(components)
    scores = prep.inverse_transform_scores(scores)

    # Reconstructed data has the same dimensions as the original data
    assert_expected_dims(data, reconstructed, policy="all")
    assert_expected_dims(data, components, policy="feature")
    assert_expected_dims(data, scores, policy="sample")

    # Reconstructed data has the same coordinates as the original data
    assert_expected_coords(data, reconstructed, policy="all")
    assert_expected_coords(data, components, policy="feature")
    assert_expected_coords(data, scores, policy="sample")

    # Reconstructed data and original data have NaNs in the same FEATURES
    # Note: NaNs in the same place is not guaranteed, since isolated NaNs will be propagated
    # to all samples in the same feature
    features_with_nans_before = data.isnull().any(sample_dims)
    features_with_nans_after = reconstructed.isnull().any(sample_dims)
    assert features_with_nans_before.equals(features_with_nans_after)

    # Reconstructed data has MultiIndex if and only if original data has MultiIndex
    data_has_multiindex_before = data_has_multiindex(data)
    data_has_multiindex_after = data_has_multiindex(reconstructed)
    assert data_has_multiindex_before == data_has_multiindex_after

    # Reconstructed data is dask if and only if original data is dask
    data_is_dask_before = data_is_dask(data)
    data_is_dask_after = data_is_dask(reconstructed)
    assert data_is_dask_before == data_is_dask_after
