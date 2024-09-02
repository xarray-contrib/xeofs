import numpy as np
import pytest
import xarray as xr

from xeofs.single import ExtendedEOF


def test_init():
    """Tests the initialization of the ExtendedEOF class"""
    eof = ExtendedEOF(n_modes=5, tau=2, embedding=2)

    # Assert preprocessor has been initialized
    assert hasattr(eof, "_params")
    assert hasattr(eof, "preprocessor")


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_fit(dim, mock_data_array):
    """Tests the fit method of the ExtendedEOF class"""

    eof = ExtendedEOF(n_modes=5, tau=2, embedding=2)
    eof.fit(mock_data_array, dim)

    # Assert the required attributes have been set
    assert hasattr(eof, "preprocessor")
    assert hasattr(eof, "data")


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_singular_values(dim, mock_data_array):
    """Tests the singular_values method of the ExtendedEOF class"""

    eof = ExtendedEOF(n_modes=5, tau=2, embedding=2)
    eof.fit(mock_data_array, dim)

    # Test singular_values method
    singular_values = eof.singular_values()
    assert isinstance(singular_values, xr.DataArray)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_explained_variance(dim, mock_data_array):
    """Tests the explained_variance method of the ExtendedEOF class"""
    eof = ExtendedEOF(n_modes=5, tau=2, embedding=2)
    eof.fit(mock_data_array, dim)

    # Test explained_variance method
    explained_variance = eof.explained_variance()
    assert isinstance(explained_variance, xr.DataArray)
    # Explained variance must be positive
    assert (explained_variance > 0).all()


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_explained_variance_ratio(dim, mock_data_array):
    """Tests the explained_variance_ratio method of the ExtendedEOF class"""
    eof = ExtendedEOF(n_modes=5, tau=2, embedding=2)
    eof.fit(mock_data_array, dim)

    # Test explained_variance_ratio method
    explained_variance_ratio = eof.explained_variance_ratio()
    assert isinstance(explained_variance_ratio, xr.DataArray)
    # Explained variance ratio must be positive
    assert (
        explained_variance_ratio > 0
    ).all(), "The explained variance ratio must be positive"
    # The sum of the explained variance ratio must be <= 1
    assert (
        explained_variance_ratio.sum() <= 1 + 1e-5
    ), "The sum of the explained variance ratio must be <= 1"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_isolated_nans(dim, mock_data_array_isolated_nans):
    """Tests the components method of the ExtendedEOF class"""
    eof = ExtendedEOF(n_modes=5, tau=2, embedding=2)
    with pytest.raises(ValueError):
        eof.fit(mock_data_array_isolated_nans, dim)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components(dim, mock_data_array):
    """Tests the components method of the ExtendedEOF class"""
    eof = ExtendedEOF(n_modes=5, tau=2, embedding=2)
    eof.fit(mock_data_array, dim)

    # Test components method
    components = eof.components()
    feature_dims = tuple(set(mock_data_array.dims) - set(dim))
    assert isinstance(components, xr.DataArray), "Components is not a DataArray"
    given_dims = set(components.dims)
    expected_dims = set(feature_dims + ("mode", "embedding"))
    assert (
        given_dims == expected_dims
    ), "Components does not have the right feature dimensions"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components_fulldim_nans(dim, mock_data_array_full_dimensional_nans):
    """Tests the components method of the ExtendedEOF class"""
    eof = ExtendedEOF(n_modes=5, tau=2, embedding=2)
    eof.fit(mock_data_array_full_dimensional_nans, dim)

    # Test components method
    components = eof.components()
    feature_dims = tuple(set(mock_data_array_full_dimensional_nans.dims) - set(dim))
    assert isinstance(components, xr.DataArray), "Components is not a DataArray"
    given_dims = set(components.dims)
    expected_dims = set(feature_dims + ("mode", "embedding"))
    assert (
        given_dims == expected_dims
    ), "Components does not have the right feature dimensions"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components_boundary_nans(dim, mock_data_array_boundary_nans):
    """Tests the components method of the ExtendedEOF class"""
    eof = ExtendedEOF(n_modes=5, tau=2, embedding=2)
    eof.fit(mock_data_array_boundary_nans, dim)

    # Test components method
    components = eof.components()
    feature_dims = tuple(set(mock_data_array_boundary_nans.dims) - set(dim))
    assert isinstance(components, xr.DataArray), "Components is not a DataArray"
    given_dims = set(components.dims)
    expected_dims = set(feature_dims + ("mode", "embedding"))
    assert (
        given_dims == expected_dims
    ), "Components does not have the right feature dimensions"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components_dataset(dim, mock_dataset):
    """Tests the components method of the ExtendedEOF class"""
    eof = ExtendedEOF(n_modes=5, tau=2, embedding=2)
    eof.fit(mock_dataset, dim)

    # Test components method
    components = eof.components()
    feature_dims = tuple(set(mock_dataset.dims) - set(dim))
    assert isinstance(components, xr.Dataset), "Components is not a Dataset"
    assert set(components.data_vars) == set(
        mock_dataset.data_vars
    ), "Components does not have the same data variables as the input Dataset"
    given_dims = set(components.dims)
    expected_dims = set(feature_dims + ("mode", "embedding"))
    assert (
        given_dims == expected_dims
    ), "Components does not have the right feature dimensions"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components_dataarray_list(dim, mock_data_array_list):
    """Tests the components method of the ExtendedEOF class"""
    eof = ExtendedEOF(n_modes=5, tau=2, embedding=2)
    eof.fit(mock_data_array_list, dim)

    # Test components method
    components = eof.components()
    feature_dims = [tuple(set(data.dims) - set(dim)) for data in mock_data_array_list]
    assert isinstance(components, list), "Components is not a list"
    assert len(components) == len(
        mock_data_array_list
    ), "Components does not have the same length as the input list"
    assert isinstance(
        components[0], xr.DataArray
    ), "Components is not a list of DataArrays"
    for comp, feat_dims in zip(components, feature_dims):
        given_dims = set(comp.dims)
        expected_dims = set(feat_dims + ("mode", "embedding"))
        assert (
            given_dims == expected_dims
        ), "Components does not have the right feature dimensions"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_scores(dim, mock_data_array):
    """Tests the scores method of the ExtendedEOF class"""
    eof = ExtendedEOF(n_modes=5, tau=2, embedding=2)
    eof.fit(mock_data_array, dim)

    # Test scores method
    scores = eof.scores()
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
def test_scores_fulldim_nans(dim, mock_data_array_full_dimensional_nans):
    """Tests the scores method of the ExtendedEOF class"""
    eof = ExtendedEOF(n_modes=5, tau=2, embedding=2)
    eof.fit(mock_data_array_full_dimensional_nans, dim)

    # Test scores method
    scores = eof.scores()
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
def test_scores_boundary_nans(dim, mock_data_array_boundary_nans):
    """Tests the scores method of the ExtendedEOF class"""
    eof = ExtendedEOF(n_modes=5, tau=2, embedding=2)
    eof.fit(mock_data_array_boundary_nans, dim)

    # Test scores method
    scores = eof.scores()
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
def test_scores_dataset(dim, mock_dataset):
    """Tests the scores method of the ExtendedEOF class"""
    eof = ExtendedEOF(n_modes=5, tau=2, embedding=2)
    eof.fit(mock_dataset, dim)

    # Test scores method
    scores = eof.scores()
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
def test_scores_dataarray_list(dim, mock_data_array_list):
    """Tests the scores method of the ExtendedEOF class"""
    eof = ExtendedEOF(n_modes=5, tau=2, embedding=2)
    eof.fit(mock_data_array_list, dim)

    # Test scores method
    scores = eof.scores()
    assert isinstance(scores, xr.DataArray)
    assert set(scores.dims) == set(
        (dim + ("mode",))
    ), "Scores does not have the right dimensions"


def test_get_params():
    """Tests the get_params method of the ExtendedEOF class"""
    eof = ExtendedEOF(n_modes=5, tau=2, embedding=2)

    # Test get_params method
    params = eof.get_params()
    assert isinstance(params, dict)
    assert params.get("n_modes") == 5
    assert params.get("tau") == 2
    assert params.get("embedding") == 2
    assert params.get("solver") == "auto"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_transform(dim, mock_data_array):
    """Test projecting new unseen data onto the components (EOFs/eigenvectors)"""

    # Create a xarray DataArray with random data
    model = ExtendedEOF(n_modes=5, tau=2, embedding=2, solver="full")
    model.fit(mock_data_array, dim)
    model.scores()

    # Create a new xarray DataArray with random data
    new_data = mock_data_array

    with pytest.raises(NotImplementedError):
        model.transform(new_data)

    # # Check that the projection has the right dimensions
    # assert projections.dims == scores.dims, "Projection has wrong dimensions"  # type: ignore

    # # Check that the projection has the right data type
    # assert isinstance(projections, xr.DataArray), "Projection is not a DataArray"

    # # Check that the projection has the right name
    # assert projections.name == "scores", "Projection has wrong name: {}".format(
    #     projections.name
    # )

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
def test_inverse_transform(dim, mock_data_array):
    """Test inverse_transform method in ExtendedEOF class."""

    # instantiate the ExtendedEOF class with necessary parameters
    eeof = ExtendedEOF(n_modes=5, tau=2, embedding=2)

    # fit the ExtendedEOF model
    eeof.fit(mock_data_array, dim=dim)
    scores = eeof.scores()

    # Test with scalar
    mode = 1
    reconstructed_data = eeof.inverse_transform(scores.sel(mode=mode))
    assert isinstance(reconstructed_data, xr.DataArray)

    # Test with slice
    mode = slice(1, 2)
    reconstructed_data = eeof.inverse_transform(scores.sel(mode=mode))
    assert isinstance(reconstructed_data, xr.DataArray)

    # Test with array of tick labels
    mode = np.array([1, 3])
    reconstructed_data = eeof.inverse_transform(scores.sel(mode=mode))
    assert isinstance(reconstructed_data, xr.DataArray)

    # Check that the reconstructed data has the same dimensions as the original data
    assert set(reconstructed_data.dims) == set(mock_data_array.dims)
