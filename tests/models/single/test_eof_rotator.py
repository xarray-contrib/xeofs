import numpy as np
import pytest
import xarray as xr

from xeofs.data_container import DataContainer
from xeofs.single import EOF, EOFRotator

from ...utilities import data_is_dask


@pytest.fixture
def eof_model(mock_data_array, dim):
    eof = EOF(n_modes=5)
    eof.fit(mock_data_array, dim)
    return eof


@pytest.fixture
def eof_model_delayed(mock_dask_data_array, dim):
    eof = EOF(n_modes=5, compute=False, check_nans=False)
    eof.fit(mock_dask_data_array, dim)
    return eof


def test_init():
    # Instantiate the EOFRotator class
    eof_rotator = EOFRotator(n_modes=3, power=2, max_iter=100, rtol=1e-6)

    assert eof_rotator._params["n_modes"] == 3
    assert eof_rotator._params["power"] == 2
    assert eof_rotator._params["max_iter"] == 100
    assert eof_rotator._params["rtol"] == 1e-6


@pytest.mark.parametrize(
    "dim",
    [
        ("time"),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_fit(eof_model):
    eof_rotator = EOFRotator(n_modes=3)
    eof_rotator.fit(eof_model)

    assert hasattr(
        eof_rotator, "model"
    ), 'The attribute "model" should be populated after fitting.'
    assert hasattr(
        eof_rotator, "data"
    ), 'The attribute "data" should be populated after fitting.'
    assert isinstance(eof_rotator.model, EOF)
    assert isinstance(eof_rotator.data, DataContainer)


@pytest.mark.parametrize(
    "dim",
    [
        ("time"),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_transform(eof_model, mock_data_array):
    eof_rotator = EOFRotator(n_modes=3)
    eof_rotator.fit(eof_model)
    projections = eof_rotator.transform(mock_data_array)

    assert isinstance(projections, xr.DataArray)


@pytest.mark.parametrize(
    "dim",
    [
        ("time"),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_inverse_transform(eof_model):
    eof_rotator = EOFRotator(n_modes=3)
    eof_rotator.fit(eof_model)
    scores = eof_rotator.data["scores"].sel(mode=1)
    Xrec = eof_rotator.inverse_transform(scores)

    assert isinstance(Xrec, xr.DataArray)


@pytest.mark.parametrize(
    "dim",
    [
        ("time"),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_explained_variance(eof_model):
    eof_rotator = EOFRotator(n_modes=3)
    eof_rotator.fit(eof_model)
    exp_var = eof_rotator.explained_variance()
    exp_var_ref = eof_model.explained_variance().sel(mode=slice(1, 3))

    assert isinstance(exp_var, xr.DataArray)
    # 3 modes should be returned
    assert exp_var.size == 3
    # The explained variance should be positive
    assert (exp_var > 0).all()
    # The sum of the explained variance should be the same
    # before and after rotation
    xr.testing.assert_allclose(exp_var.sum(), exp_var_ref.sum())


@pytest.mark.parametrize(
    "dim",
    [
        ("time"),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_explained_variance_ratio(eof_model):
    eof_rotator = EOFRotator(n_modes=3)
    eof_rotator.fit(eof_model)
    exp_var_ratio = eof_rotator.explained_variance_ratio()
    exp_var_ratio_ref = eof_model.explained_variance_ratio().sel(mode=slice(1, 3))

    assert isinstance(exp_var_ratio, xr.DataArray)
    # 3 modes should be returned
    assert exp_var_ratio.size == 3
    # The explained variance should be positive
    assert (exp_var_ratio > 0).all()
    # The total of the explained variance ratio should be <= 1
    assert exp_var_ratio.sum() <= 1
    # The sum of the explained variance should be the same
    # before and after rotation
    xr.testing.assert_allclose(exp_var_ratio.sum(), exp_var_ratio_ref.sum())


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components(eof_model):
    eof_rotator = EOFRotator(n_modes=3)
    eof_rotator.fit(eof_model)
    components = eof_rotator.components()

    assert isinstance(components, xr.DataArray)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_scores(eof_model):
    eof_rotator = EOFRotator(n_modes=3)
    eof_rotator.fit(eof_model)
    scores = eof_rotator.scores()

    assert isinstance(scores, xr.DataArray)


@pytest.mark.parametrize(
    "dim, compute",
    [
        (("time",), True),
        (("lat", "lon"), True),
        (("lon", "lat"), True),
        (("time",), False),
        (("lat", "lon"), False),
        (("lon", "lat"), False),
    ],
)
def test_compute(eof_model_delayed, compute):
    eof_rotator = EOFRotator(n_modes=5, compute=compute, max_iter=20, rtol=1e-4)
    eof_rotator.fit(eof_model_delayed)

    if compute:
        assert not data_is_dask(eof_rotator.data["explained_variance"])
        assert not data_is_dask(eof_rotator.data["components"])
        assert not data_is_dask(eof_rotator.data["rotation_matrix"])

    else:
        assert data_is_dask(eof_rotator.data["explained_variance"])
        assert data_is_dask(eof_rotator.data["components"])
        assert data_is_dask(eof_rotator.data["rotation_matrix"])


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
    original_unrotated = EOF()
    original_unrotated.fit(mock_data_array, dim)

    original = EOFRotator()
    original.fit(original_unrotated)

    # Save the EOF model
    original.save(tmp_path / "eof", engine=engine)

    # Check that the EOF model has been saved
    assert (tmp_path / "eof").exists()

    # Recreate the model from saved file
    loaded = EOFRotator.load(tmp_path / "eof", engine=engine)

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
    rotator = EOFRotator()
    rotator.fit(model)
    dt = rotator.serialize()
    rebuilt_rotator = EOFRotator.deserialize(dt)
    assert np.allclose(
        rotator.transform(mock_data_array), rebuilt_rotator.transform(mock_data_array)
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
    rotator = EOFRotator()
    rotator.fit(model)
    dt = rotator.serialize()
    rebuilt_rotator = EOFRotator.deserialize(dt)
    assert np.allclose(
        rotator.transform(mock_dataset), rebuilt_rotator.transform(mock_dataset)
    )
