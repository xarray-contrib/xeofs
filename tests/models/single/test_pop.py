import numpy as np
import pytest
import xarray as xr

from xeofs.single import POP


def test_init():
    """Tests the initialization of the POP class"""
    pop = POP(n_modes=5, standardize=True, use_coslat=True)

    # Assert preprocessor has been initialized
    assert hasattr(pop, "_params")
    assert hasattr(pop, "preprocessor")
    assert hasattr(pop, "whitener")


def test_fit(mock_data_array):
    pop = POP()
    pop.fit(mock_data_array, "time")


def test_eigenvalues(mock_data_array):
    pop = POP()
    pop.fit(mock_data_array, "time")

    eigenvalues = pop.eigenvalues()
    assert isinstance(eigenvalues, xr.DataArray)


def test_damping_times(mock_data_array):
    pop = POP()
    pop.fit(mock_data_array, "time")

    times = pop.damping_times()
    assert isinstance(times, xr.DataArray)


def test_periods(mock_data_array):
    pop = POP()
    pop.fit(mock_data_array, "time")

    periods = pop.periods()
    assert isinstance(periods, xr.DataArray)


def test_components(mock_data_array):
    """Tests the components method of the POP class"""
    sample_dim = ("time",)
    pop = POP()
    pop.fit(mock_data_array, sample_dim)

    # Test components method
    components = pop.components()
    feature_dims = tuple(set(mock_data_array.dims) - set(sample_dim))
    assert isinstance(components, xr.DataArray), "Components is not a DataArray"
    assert set(components.dims) == set(
        ("mode",) + feature_dims
    ), "Components does not have the right feature dimensions"


def test_scores(mock_data_array):
    """Tests the scores method of the POP class"""
    sample_dim = ("time",)
    pop = POP()
    pop.fit(mock_data_array, sample_dim)

    # Test scores method
    scores = pop.scores()
    assert isinstance(scores, xr.DataArray), "Scores is not a DataArray"
    assert set(scores.dims) == set(
        (sample_dim + ("mode",))
    ), "Scores does not have the right dimensions"


def test_transform(mock_data_array):
    """Test projecting new unseen data onto the POPs"""
    dim = ("time",)
    data = mock_data_array.isel({dim[0]: slice(1, None)})
    new_data = mock_data_array.isel({dim[0]: slice(0, 1)})

    # Create a xarray DataArray with random data
    model = POP(n_modes=2, solver="full")
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


def test_inverse_transform(mock_data_array):
    """Test inverse_transform method in POP class."""

    dim = ("time",)
    # instantiate the POP class with necessary parameters
    pop = POP(n_modes=20, standardize=True)

    # fit the POP model
    pop.fit(mock_data_array, dim=dim)
    scores = pop.scores()

    # Test with single mode
    scores_selection = scores.sel(mode=1)
    X_rec_1 = pop.inverse_transform(scores_selection)
    assert isinstance(X_rec_1, xr.DataArray)

    # Test with single mode as list
    scores_selection = scores.sel(mode=[1])
    X_rec_1_list = pop.inverse_transform(scores_selection)
    assert isinstance(X_rec_1_list, xr.DataArray)

    # Single mode and list should be equal
    xr.testing.assert_allclose(X_rec_1, X_rec_1_list)

    # Test with all modes
    X_rec = pop.inverse_transform(scores)
    assert isinstance(X_rec, xr.DataArray)

    # Check that the reconstructed data has the same dimensions as the original data
    assert set(X_rec.dims) == set(mock_data_array.dims)


@pytest.mark.parametrize("engine", ["zarr"])
def test_save_load(mock_data_array, tmp_path, engine):
    """Test save/load methods in POP class, ensuring that we can
    roundtrip the model and get the same results when transforming
    data."""
    # NOTE: netcdf4 does not support complex data types, so we use only zarr here
    dim = "time"
    original = POP()
    original.fit(mock_data_array, dim)

    # Save the POP model
    original.save(tmp_path / "pop", engine=engine)

    # Check that the POP model has been saved
    assert (tmp_path / "pop").exists()

    # Recreate the model from saved file
    loaded = POP.load(tmp_path / "pop", engine=engine)

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


def test_serialize_deserialize_dataarray(mock_data_array):
    """Test roundtrip serialization when the model is fit on a DataArray."""
    dim = "time"
    model = POP()
    model.fit(mock_data_array, dim)
    dt = model.serialize()
    rebuilt_model = POP.deserialize(dt)
    assert np.allclose(
        model.transform(mock_data_array), rebuilt_model.transform(mock_data_array)
    )


def test_serialize_deserialize_dataset(mock_dataset):
    """Test roundtrip serialization when the model is fit on a Dataset."""
    dim = "time"
    model = POP()
    model.fit(mock_dataset, dim)
    dt = model.serialize()
    rebuilt_model = POP.deserialize(dt)
    assert np.allclose(
        model.transform(mock_dataset), rebuilt_model.transform(mock_dataset)
    )
