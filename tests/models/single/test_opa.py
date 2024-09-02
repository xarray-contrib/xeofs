import numpy as np
import pytest
import xarray as xr

from xeofs.single import OPA


@pytest.fixture
def opa_model():
    return OPA(n_modes=3, tau_max=3, n_pca_modes=19)


def test_init():
    """Tests the initialization of the OPA class"""
    opa = OPA(n_modes=3, tau_max=3, n_pca_modes=19, use_coslat=True)

    # Assert preprocessor has been initialized
    assert hasattr(opa, "_params")
    assert hasattr(opa, "preprocessor")


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_fit(dim, mock_data_array, opa_model):
    """Tests the fit method of the OPA class"""

    opa_model.fit(mock_data_array, dim)

    # Assert the required attributes have been set
    assert hasattr(opa_model, "preprocessor")
    assert hasattr(opa_model, "data")


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_decorrelation_time(dim, mock_data_array, opa_model):
    """Tests the decorrelation_time method of the OPA class"""

    opa_model.fit(mock_data_array, dim)

    # Test decorrelation time method
    decorrelation_time = opa_model.decorrelation_time()
    assert isinstance(decorrelation_time, xr.DataArray)

    # decorrelation times are all positive
    assert (decorrelation_time > 0).all()


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components(dim, mock_data_array, opa_model):
    """Tests the components method of the OPA class"""
    opa_model.fit(mock_data_array, dim)

    # Test components method
    components = opa_model.components()
    feature_dims = tuple(set(mock_data_array.dims) - set(dim))
    assert isinstance(components, xr.DataArray), "Components is not a DataArray"
    assert set(components.dims) == set(
        ("mode",) + feature_dims
    ), "Components does not have the right feature dimensions"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_filter_patterns(dim, mock_data_array, opa_model):
    """Tests the filter_patterns method of the OPA class"""
    opa_model.fit(mock_data_array, dim)

    # Test components method
    components = opa_model.filter_patterns()
    feature_dims = tuple(set(mock_data_array.dims) - set(dim))
    assert isinstance(components, xr.DataArray), "Components is not a DataArray"
    assert set(components.dims) == set(
        ("mode",) + feature_dims
    ), "Components does not have the right feature dimensions"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components_dataset(dim, mock_dataset, opa_model):
    """Tests the components method of the OPA class"""
    opa_model.fit(mock_dataset, dim)

    # Test components method
    components = opa_model.components()
    feature_dims = tuple(set(mock_dataset.dims) - set(dim))
    assert isinstance(components, xr.Dataset), "Components is not a Dataset"
    assert set(components.data_vars) == set(
        mock_dataset.data_vars
    ), "Components does not have the same data variables as the input Dataset"
    assert set(components.dims) == set(
        ("mode",) + feature_dims
    ), "Components does not have the right feature dimensions"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components_dataarray_list(dim, mock_data_array_list, opa_model):
    """Tests the components method of the OPA class"""
    opa_model.fit(mock_data_array_list, dim)

    # Test components method
    components = opa_model.components()
    feature_dims = [tuple(set(data.dims) - set(dim)) for data in mock_data_array_list]
    assert isinstance(components, list), "Components is not a list"
    assert len(components) == len(
        mock_data_array_list
    ), "Components does not have the same length as the input list"
    assert isinstance(
        components[0], xr.DataArray
    ), "Components is not a list of DataArrays"
    for comp, feat_dims in zip(components, feature_dims):
        assert set(comp.dims) == set(
            ("mode",) + feat_dims
        ), "Components does not have the right feature dimensions"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_scores(dim, mock_data_array, opa_model):
    """Tests the scores method of the OPA class"""
    opa_model.fit(mock_data_array, dim)

    # Test scores method
    scores = opa_model.scores()
    assert isinstance(scores, xr.DataArray), "Scores is not a DataArray"
    assert set(scores.dims) == set(
        (dim + ("mode",))
    ), "Scores does not have the right dimensions"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_scores_dataset(dim, mock_dataset, opa_model):
    """Tests the scores method of the OPA class"""
    opa_model.fit(mock_dataset, dim)

    # Test scores method
    scores = opa_model.scores()
    assert isinstance(scores, xr.DataArray)
    assert set(scores.dims) == set(
        (dim + ("mode",))
    ), "Scores does not have the right dimensions"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_scores_dataarray_list(dim, mock_data_array_list, opa_model):
    """Tests the scores method of the OPA class"""
    opa_model.fit(mock_data_array_list, dim)

    # Test scores method
    scores = opa_model.scores()
    assert isinstance(scores, xr.DataArray)
    assert set(scores.dims) == set(
        (dim + ("mode",))
    ), "Scores does not have the right dimensions"


def test_get_params(opa_model):
    """Tests the get_params method of the OPA class"""
    # Test get_params method
    params = opa_model.get_params()
    assert isinstance(params, dict)
    assert params.get("n_modes") == 3
    assert params.get("tau_max") == 3
    assert params.get("n_pca_modes") == 19
    assert params.get("standardize") is False
    assert params.get("use_coslat") is False
    assert params.get("solver") == "auto"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_transform(dim, mock_data_array, opa_model):
    """Transform is not implemented for OPA"""

    # Create a xarray DataArray with random data
    opa_model.fit(mock_data_array, dim)

    with pytest.raises(NotImplementedError):
        opa_model.transform(mock_data_array)
    # scores = opa_model.scores()

    # # Create a new xarray DataArray with random data
    # new_data = mock_data_array

    # projections = opa_model.transform(new_data)

    # # Check that the projection has the right dimensions
    # assert projections.dims == scores.dims, "Projection has wrong dimensions"  # type: ignore

    # # Check that the projection has the right data type
    # assert isinstance(projections, xr.DataArray), "Projection is not a DataArray"

    # # Check that the projection has the right name
    # assert projections.name == "scores", "Projection has wrong name"

    # # Check that the projection's data is the same as the scores
    # np.testing.assert_allclose(
    #     scores.sel(mode=slice(1, 3)), projections.sel(mode=slice(1, 3)), rtol=1e-3
    # )


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_inverse_transform(dim, mock_data_array, opa_model):
    """Test inverse_transform method in EOF class."""

    # fit the EOF model
    opa_model.fit(mock_data_array, dim=dim)
    scores = opa_model.scores()

    with pytest.raises(NotImplementedError):
        opa_model.inverse_transform(scores)

    # # Test with scalar
    # mode = 1
    # reconstructed_data = opa_model.inverse_transform(mode)
    # assert isinstance(reconstructed_data, xr.DataArray)

    # # Test with slice
    # mode = slice(1, 2)
    # reconstructed_data = opa_model.inverse_transform(mode)
    # assert isinstance(reconstructed_data, xr.DataArray)

    # # Test with array of tick labels
    # mode = np.array([1, 3])
    # reconstructed_data = opa_model.inverse_transform(mode)
    # assert isinstance(reconstructed_data, xr.DataArray)

    # # Check that the reconstructed data has the same dimensions as the original data
    # assert set(reconstructed_data.dims) == set(mock_data_array.dims)


# Check mathematical properties
# =============================================================================
@pytest.mark.parametrize(
    "dim, use_coslat",
    [
        (("time",), True),
        (("lat", "lon"), False),
        (("lon", "lat"), False),
    ],
)
def test_U_orthogonal(dim, use_coslat, mock_data_array):
    """U patterns are orthogonal"""
    model = OPA(
        n_modes=3, tau_max=3, n_pca_modes=19, standardize=True, use_coslat=use_coslat
    )
    model.fit(mock_data_array, dim=dim)
    U = model._U.values
    assert np.allclose(
        U.T @ U, np.eye(U.shape[1]), atol=1e-5
    ), "U patterns are not orthogonal"


# IGNORE THIS TEST: Current implementation yields C0 in PCA space and FP in original space
# therefore the test fails
# @pytest.mark.parametrize(
#     "dim, use_coslat",
#     [
#         (("time",), True),
#         (("lat", "lon"), False),
#         (("lon", "lat"), False),
#     ],
# )
# def test_filter_patterns_biorthogonal(dim, use_coslat, mock_data_array):
#     """Filter patterns are biorthogonal"""
#     n_pca_modes = 20
#     model = OPA(
#         n_modes=3,
#         tau_max=3,
#         n_pca_modes=n_pca_modes,
#         standardize=True,
#         use_coslat=use_coslat,
#     )
#     model.fit(mock_data_array, dim=dim)
#     FP = model.data.filter_patterns.values
#     C0 = model._C0.values  # zero-lag covariance matrix
#     assert C0.shape == (n_pca_modes, n_pca_modes)
#     check = FP.T @ C0 @ FP
#     assert np.allclose(
#         check, np.eye(check.shape[1]), atol=1e-5
#     ), "Filter patterns are are not biorthogonal"


@pytest.mark.parametrize(
    "dim, use_coslat",
    [
        (("time",), True),
        (("lat", "lon"), False),
        (("lon", "lat"), False),
    ],
)
def test_scores_uncorrelated(dim, use_coslat, mock_data_array):
    """Scores are uncorrelated"""
    n_pca_modes = 20
    model = OPA(
        n_modes=3,
        tau_max=3,
        n_pca_modes=n_pca_modes,
        standardize=True,
        use_coslat=use_coslat,
    )
    model.fit(mock_data_array, dim=dim)
    scores = model.data["scores"].values
    check = scores.T @ scores / (scores.shape[0] - 1)
    assert np.allclose(
        check, np.eye(check.shape[1]), atol=1e-5
    ), "Scores are not uncorrelated"
