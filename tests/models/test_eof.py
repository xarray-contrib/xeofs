from copy import deepcopy

import numpy as np
import xarray as xr
import pytest
from dask.array import Array as DaskArray  # type: ignore
from numpy.testing import assert_allclose

from xeofs.models.eof import EOF


def test_init():
    """Tests the initialization of the EOF class"""
    eof = EOF(n_modes=5, standardize=True, use_coslat=True)

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
    """Tests the fit method of the EOF class"""

    eof = EOF()
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
    """Tests the singular_values method of the EOF class"""

    eof = EOF()
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
    """Tests the explained_variance method of the EOF class"""
    eof = EOF()
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
    """Tests the explained_variance_ratio method of the EOF class"""
    eof = EOF()
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
    """Tests the components method of the EOF class"""
    eof = EOF()
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
    """Tests the components method of the EOF class"""
    eof = EOF()
    eof.fit(mock_data_array, dim)

    # Test components method
    components = eof.components()
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
    """Tests the components method of the EOF class"""
    eof = EOF()
    eof.fit(mock_data_array_full_dimensional_nans, dim)

    # Test components method
    components = eof.components()
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
    """Tests the components method of the EOF class"""
    eof = EOF()
    eof.fit(mock_data_array_boundary_nans, dim)

    # Test components method
    components = eof.components()
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
    """Tests the components method of the EOF class"""
    eof = EOF()
    eof.fit(mock_dataset, dim)

    # Test components method
    components = eof.components()
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
    """Tests the components method of the EOF class"""
    eof = EOF()
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
    """Tests the scores method of the EOF class"""
    eof = EOF()
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
    """Tests the scores method of the EOF class"""
    eof = EOF()
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
    """Tests the scores method of the EOF class"""
    eof = EOF()
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
    """Tests the scores method of the EOF class"""
    eof = EOF()
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
    """Tests the scores method of the EOF class"""
    eof = EOF()
    eof.fit(mock_data_array_list, dim)

    # Test scores method
    scores = eof.scores()
    assert isinstance(scores, xr.DataArray)
    assert set(scores.dims) == set(
        (dim + ("mode",))
    ), "Scores does not have the right dimensions"


def test_get_params():
    """Tests the get_params method of the EOF class"""
    eof = EOF(n_modes=5, standardize=True, use_coslat=True)

    # Test get_params method
    params = eof.get_params()
    assert isinstance(params, dict)
    assert params.get("n_modes") == 5
    assert params.get("standardize") is True
    assert params.get("use_coslat") is True
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
    data = mock_data_array.isel({dim[0]: slice(1, None)})
    new_data = mock_data_array.isel({dim[0]: slice(0, 1)})

    # Create a xarray DataArray with random data
    model = EOF(n_modes=2, solver="full")
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
    """Test projecting new unseen data onto the components (EOFs/eigenvectors)"""
    data = mock_data_array.isel()

    # Create a xarray DataArray with random data
    model = EOF(n_modes=2, solver="full")
    model.fit(data, dim)

    # Set a new feature to NaN and attempt to project data onto the components
    feature_dims = list(set(data.dims) - set(dim))
    data_missing = data.copy()
    data_missing.loc[{feature_dims[0]: data[feature_dims[0]][0].values}] = np.nan

    # with nan checking, transform should fail if any new features are NaN
    with pytest.raises(ValueError):
        model.transform(data_missing)

    # without nan checking, transform will succeed but be all nan
    model = EOF(n_modes=2, solver="full", check_nans=False)
    model.fit(data, dim)

    data_transformed = model.transform(data_missing)
    assert data_transformed.isnull().all()


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_inverse_transform(dim, mock_data_array):
    """Test inverse_transform method in EOF class."""

    # instantiate the EOF class with necessary parameters
    eof = EOF(n_modes=3, standardize=True)

    # fit the EOF model
    eof.fit(mock_data_array, dim=dim)

    # Test with single mode
    scores = eof.data["scores"].sel(mode=1)
    reconstructed_data = eof.inverse_transform(scores)
    assert isinstance(reconstructed_data, xr.DataArray)

    # Test with all modes
    scores = eof.data["scores"]
    reconstructed_data = eof.inverse_transform(scores)
    assert isinstance(reconstructed_data, xr.DataArray)

    # Check that the reconstructed data has the same dimensions as the original data
    assert set(reconstructed_data.dims) == set(mock_data_array.dims)


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
    """Test save/load methods in EOF class, ensuring that we can
    roundtrip the model and get the same results when transforming
    data."""
    original = EOF()
    original.fit(mock_data_array, dim)

    # Save the EOF model
    original.save(tmp_path / "eof", engine=engine)

    # Check that the EOF model has been saved
    assert (tmp_path / "eof").exists()

    # Recreate the model from saved file
    loaded = EOF.load(tmp_path / "eof", engine=engine)

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
    model = EOF()
    model.fit(mock_data_array, dim)
    dt = model.serialize()
    rebuilt_model = EOF.deserialize(dt)
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
    model = EOF()
    model.fit(mock_dataset, dim)
    dt = model.serialize()
    rebuilt_model = EOF.deserialize(dt)
    assert np.allclose(
        model.transform(mock_dataset), rebuilt_model.transform(mock_dataset)
    )
