import numpy as np
import pytest
import xarray as xr

from xeofs.single import SparsePCA


def test_init():
    """Tests the initialization of the SparsePCA class"""
    spca = SparsePCA(n_modes=5, standardize=True, use_coslat=True)

    # Assert preprocessor has been initialized
    assert hasattr(spca, "_params")
    assert hasattr(spca, "preprocessor")


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_fit(dim, mock_data_array):
    """Tests the fit method of the SparsePCA class"""

    spca = SparsePCA()
    spca.fit(mock_data_array, dim)

    # Assert the required attributes have been set
    assert hasattr(spca, "preprocessor")
    assert hasattr(spca, "data")


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_explained_variance(dim, mock_data_array):
    """Tests the explained_variance method of the SparsePCA class"""
    spca = SparsePCA()
    spca.fit(mock_data_array, dim)

    # Test explained_variance method
    explained_variance = spca.explained_variance()
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
    """Tests the explained_variance_ratio method of the SparsePCA class"""
    spca = SparsePCA()
    spca.fit(mock_data_array, dim)

    # Test explained_variance_ratio method
    explained_variance_ratio = spca.explained_variance_ratio()
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
    """Tests the components method of the SparsePCA class"""
    spca = SparsePCA()
    with pytest.raises(ValueError):
        spca.fit(mock_data_array_isolated_nans, dim)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components(dim, mock_data_array):
    """Tests the components method of the SparsePCA class"""
    spca = SparsePCA()
    spca.fit(mock_data_array, dim)

    # Test components method
    components = spca.components()
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
def test_components_fulldim_nans(dim, mock_data_array_full_dimensional_nans):
    """Tests the components method of the SparsePCA class"""
    spca = SparsePCA()
    spca.fit(mock_data_array_full_dimensional_nans, dim)

    # Test components method
    components = spca.components()
    feature_dims = tuple(set(mock_data_array_full_dimensional_nans.dims) - set(dim))
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
def test_components_boundary_nans(dim, mock_data_array_boundary_nans):
    """Tests the components method of the SparsePCA class"""
    spca = SparsePCA()
    spca.fit(mock_data_array_boundary_nans, dim)

    # Test components method
    components = spca.components()
    feature_dims = tuple(set(mock_data_array_boundary_nans.dims) - set(dim))
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
def test_components_dataset(dim, mock_dataset):
    """Tests the components method of the SparsePCA class"""
    spca = SparsePCA()
    spca.fit(mock_dataset, dim)

    # Test components method
    components = spca.components()
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
def test_components_dataarray_list(dim, mock_data_array_list):
    """Tests the components method of the SparsePCA class"""
    spca = SparsePCA()
    spca.fit(mock_data_array_list, dim)

    # Test components method
    components = spca.components()
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
def test_scores(dim, mock_data_array):
    """Tests the scores method of the SparsePCA class"""
    spca = SparsePCA()
    spca.fit(mock_data_array, dim)

    # Test scores method
    scores = spca.scores()
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
    """Tests the scores method of the SparsePCA class"""
    spca = SparsePCA()
    spca.fit(mock_data_array_full_dimensional_nans, dim)

    # Test scores method
    scores = spca.scores()
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
    """Tests the scores method of the SparsePCA class"""
    spca = SparsePCA()
    spca.fit(mock_data_array_boundary_nans, dim)

    # Test scores method
    scores = spca.scores()
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
    """Tests the scores method of the SparsePCA class"""
    spca = SparsePCA()
    spca.fit(mock_dataset, dim)

    # Test scores method
    scores = spca.scores()
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
    """Tests the scores method of the SparsePCA class"""
    spca = SparsePCA()
    spca.fit(mock_data_array_list, dim)

    # Test scores method
    scores = spca.scores()
    assert isinstance(scores, xr.DataArray)
    assert set(scores.dims) == set(
        (dim + ("mode",))
    ), "Scores does not have the right dimensions"


def test_get_params():
    """Tests the get_params method of the SparsePCA class"""
    spca = SparsePCA(n_modes=5, standardize=True, use_coslat=True, alpha=0.4)

    # Test get_params method
    params = spca.get_params()
    assert isinstance(params, dict)
    assert params.get("n_modes") == 5
    assert params.get("standardize") is True
    assert params.get("use_coslat") is True
    assert params.get("alpha") == 0.4


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_transform(dim, mock_data_array):
    """Test projecting new unseen data onto the components (SparsePCAs/eigenvectors)"""
    data = mock_data_array.isel({dim[0]: slice(1, None)})
    new_data = mock_data_array.isel({dim[0]: slice(0, 1)})

    # Create a xarray DataArray with random data
    model = SparsePCA(n_modes=2, solver="full")
    model.fit(data, dim)
    scores = model.scores()

    # Project data onto the components
    projections = model.transform(data)

    # Check that the projection has the right dimensions
    assert projections.dims == scores.dims, "Projection has wrong dimensions"  # type: ignore

    # Check that the projection has the right data type
    assert isinstance(projections, xr.DataArray), "Projection is not a DataArray"

    # Check that the projection has the right name
    assert projections.name == "scores", "Projection has wrong name: {}".format(
        projections.name
    )

    # Check that the projection's data is the same as the scores
    np.testing.assert_allclose(
        scores.sel(mode=slice(1, 3)), projections.sel(mode=slice(1, 3)), rtol=1e-3
    )

    # Project unseen data onto the components
    new_projections = model.transform(new_data)

    # Check that the projection has the right dimensions
    assert new_projections.dims == scores.dims, "Projection has wrong dimensions"  # type: ignore

    # Check that the projection has the right data type
    assert isinstance(new_projections, xr.DataArray), "Projection is not a DataArray"

    # Check that the projection has the right name
    assert new_projections.name == "scores", "Projection has wrong name: {}".format(
        new_projections.name
    )

    # Ensure that the new projections are not NaNs
    assert np.all(new_projections.notnull().values), "New projections contain NaNs"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_transform_nan_feature(dim, mock_data_array):
    """Test projecting new unseen data onto the components (SparsePCAs/eigenvectors)"""
    data = mock_data_array.isel()

    # Create a xarray DataArray with random data
    model = SparsePCA(n_modes=2, solver="full")
    model.fit(data, dim)

    # Set a new feature to NaN and attempt to project data onto the components
    feature_dims = list(set(data.dims) - set(dim))
    data_missing = data.copy()
    data_missing.loc[{feature_dims[0]: data[feature_dims[0]][0].values}] = np.nan

    # with nan checking, transform should fail if any new features are NaN
    with pytest.raises(ValueError):
        model.transform(data_missing)

    # without nan checking, transform will succeed but be all nan
    model = SparsePCA(n_modes=2, solver="full", check_nans=False)
    model.fit(data, dim)

    data_transformed = model.transform(data_missing)
    assert data_transformed.isnull().all()


@pytest.mark.parametrize(
    "dim, normalized",
    [
        (("time",), True),
        (("lat", "lon"), True),
        (("lon", "lat"), True),
        (("time",), False),
        (("lat", "lon"), False),
        (("lon", "lat"), False),
    ],
)
def test_inverse_transform(dim, mock_data_array, normalized):
    """Test inverse_transform method in SparsePCA class."""

    # instantiate the SparsePCA class with necessary parameters
    spca = SparsePCA(
        n_modes=20,
        alpha=1e-5,
        beta=1e-5,
        standardize=True,
        max_iter=2000,
        tol=1e-9,
        solver="full",
    )

    # fit the SparsePCA model
    spca.fit(mock_data_array, dim=dim)
    scores = spca.scores(normalized=normalized)

    # Test with single mode
    scores_selection = scores.sel(mode=1)
    X_rec_1 = spca.inverse_transform(scores_selection)
    assert isinstance(X_rec_1, xr.DataArray)

    # Test with single mode as list
    scores_selection = scores.sel(mode=[1])
    X_rec_1_list = spca.inverse_transform(scores_selection)
    assert isinstance(X_rec_1_list, xr.DataArray)

    # Single mode and list should be equal
    xr.testing.assert_allclose(X_rec_1, X_rec_1_list)

    # Test with all modes
    X_rec = spca.inverse_transform(scores, normalized=normalized)
    assert isinstance(X_rec, xr.DataArray)

    # Check that the reconstructed data has the same dimensions as the original data
    assert set(X_rec.dims) == set(mock_data_array.dims)

    # Reconstructed data should be close to the original data
    orig_dim_order = mock_data_array.dims
    X_rec = X_rec.transpose(*orig_dim_order)
    xr.testing.assert_allclose(mock_data_array, X_rec, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
@pytest.mark.parametrize("engine", ["netcdf4", "zarr"])
def test_save_load(dim, mock_data_array, tmp_path, engine):
    """Test save/load methods in SparsePCA class, ensuring that we can
    roundtrip the model and get the same results when transforming
    data."""
    original = SparsePCA()
    original.fit(mock_data_array, dim)

    # Save the SparsePCA model
    original.save(tmp_path / "spca", engine=engine)

    # Check that the SparsePCA model has been saved
    assert (tmp_path / "spca").exists()

    # Recreate the model from saved file
    loaded = SparsePCA.load(tmp_path / "spca", engine=engine)

    # Check that the params and DataContainer objects match
    assert original.get_params() == loaded.get_params()
    assert all([key in loaded.data for key in original.data])
    for key in original.data:
        if original.data._allow_compute[key]:
            assert loaded.data[key].equals(original.data[key])
        else:
            # but ensure that input data is not saved by default
            assert loaded.data[key].size <= 1
            assert loaded.data[key].attrs["placeholder"] is True

    # Test that the recreated model can be used to transform new data
    assert np.allclose(
        original.transform(mock_data_array), loaded.transform(mock_data_array)
    )

    # The loaded model should also be able to inverse_transform new data
    assert np.allclose(
        original.inverse_transform(original.scores()),
        loaded.inverse_transform(loaded.scores()),
    )


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_serialize_deserialize_dataarray(dim, mock_data_array):
    """Test roundtrip serialization when the model is fit on a DataArray."""
    model = SparsePCA()
    model.fit(mock_data_array, dim)
    dt = model.serialize()
    rebuilt_model = SparsePCA.deserialize(dt)
    assert np.allclose(
        model.transform(mock_data_array), rebuilt_model.transform(mock_data_array)
    )


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_serialize_deserialize_dataset(dim, mock_dataset):
    """Test roundtrip serialization when the model is fit on a Dataset."""
    model = SparsePCA()
    model.fit(mock_dataset, dim)
    dt = model.serialize()
    rebuilt_model = SparsePCA.deserialize(dt)
    assert np.allclose(
        model.transform(mock_dataset), rebuilt_model.transform(mock_dataset)
    )


def test_complex_input_inverse_transform(mock_complex_data_array):
    """Test that the SparsePCA model raises an error with complex input data."""

    model = SparsePCA(n_modes=128)
    with pytest.raises(TypeError):
        model.fit(mock_complex_data_array, dim="time")
