import pytest
import numpy as np
import xarray as xr
from dask.array import Array as DaskArray  # type: ignore

from xeofs.models import EOF, EOFRotator
from xeofs.data_container.eof_rotator_data_container import (
    EOFRotatorDataContainer,
)


@pytest.fixture
def eof_model(mock_data_array, dim):
    eof = EOF(n_modes=5)
    eof.fit(mock_data_array, dim)
    return eof


@pytest.fixture
def eof_model_delayed(mock_dask_data_array, dim):
    eof = EOF(n_modes=5)
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
    assert type(eof_rotator.model) == EOF
    assert type(eof_rotator.data) == EOFRotatorDataContainer


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
    Xrec = eof_rotator.inverse_transform(mode=1)

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
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_compute(eof_model_delayed):
    eof_rotator = EOFRotator(n_modes=5)
    eof_rotator.fit(eof_model_delayed)

    # before computation, the attributes should be dask arrays
    assert isinstance(
        eof_rotator.data.explained_variance.data, DaskArray
    ), "The attribute _explained_variance should be a dask array."
    assert isinstance(
        eof_rotator.data.explained_variance_ratio.data, DaskArray
    ), "The attribute _explained_variance_ratio should be a dask array."
    assert isinstance(
        eof_rotator.data.components.data, DaskArray
    ), "The attribute _components should be a dask array."
    assert isinstance(
        eof_rotator.data.rotation_matrix.data, DaskArray
    ), "The attribute _rotation_matrix should be a dask array."
    assert isinstance(
        eof_rotator.data.scores.data, DaskArray
    ), "The attribute _scores should be a dask array."

    eof_rotator.compute()

    # after computation, the attributes should be numpy ndarrays
    assert isinstance(
        eof_rotator.data.explained_variance.data, np.ndarray
    ), "The attribute _explained_variance should be a numpy ndarray."
    assert isinstance(
        eof_rotator.data.explained_variance_ratio.data, np.ndarray
    ), "The attribute _explained_variance_ratio should be a numpy ndarray."
    assert isinstance(
        eof_rotator.data.components.data, np.ndarray
    ), "The attribute _components should be a numpy ndarray."
    assert isinstance(
        eof_rotator.data.rotation_matrix.data, np.ndarray
    ), "The attribute _rotation_matrix should be a numpy ndarray."
    assert isinstance(
        eof_rotator.data.scores.data, np.ndarray
    ), "The attribute _scores should be a numpy ndarray."
