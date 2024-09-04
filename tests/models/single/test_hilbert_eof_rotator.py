import pytest
import xarray as xr

from xeofs.data_container import DataContainer
from xeofs.single import HilbertEOF, HilbertEOFRotator


@pytest.fixture
def ceof_model(mock_data_array, dim):
    ceof = HilbertEOF(n_modes=5)
    ceof.fit(mock_data_array, dim)
    return ceof


@pytest.fixture
def ceof_model_delayed(mock_dask_data_array, dim):
    ceof = HilbertEOF(n_modes=5)
    ceof.fit(mock_dask_data_array, dim)
    return ceof


def test_init():
    # Instantiate the HilbertEOFRotator class
    ceof_rotator = HilbertEOFRotator(n_modes=3, power=2, max_iter=100, rtol=1e-6)

    assert ceof_rotator._params["n_modes"] == 3
    assert ceof_rotator._params["power"] == 2
    assert ceof_rotator._params["max_iter"] == 100
    assert ceof_rotator._params["rtol"] == 1e-6


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_fit(ceof_model):
    ceof_rotator = HilbertEOFRotator(n_modes=3)
    ceof_rotator.fit(ceof_model)

    assert hasattr(
        ceof_rotator, "model"
    ), 'The attribute "model" should be populated after fitting.'
    assert hasattr(
        ceof_rotator, "data"
    ), 'The attribute "data" should be populated after fitting.'
    assert isinstance(ceof_rotator.model, HilbertEOF)
    assert isinstance(ceof_rotator.data, DataContainer)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_transform_not_implemented(ceof_model, mock_data_array):
    ceof_rotator = HilbertEOFRotator(n_modes=3)
    ceof_rotator.fit(ceof_model)

    with pytest.raises(NotImplementedError):
        ceof_rotator.transform(mock_data_array)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_inverse_transform(ceof_model):
    ceof_rotator = HilbertEOFRotator(n_modes=3)
    ceof_rotator.fit(ceof_model)
    scores = ceof_rotator.data["scores"].isel(mode=1)
    Xrec = ceof_rotator.inverse_transform(scores)

    assert isinstance(Xrec, xr.DataArray)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components_amplitude(ceof_model):
    ceof_rotator = HilbertEOFRotator(n_modes=3)
    ceof_rotator.fit(ceof_model)
    comps_amp = ceof_rotator.components_amplitude()

    assert isinstance(comps_amp, xr.DataArray)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components_phase(ceof_model):
    ceof_rotator = HilbertEOFRotator(n_modes=3)
    ceof_rotator.fit(ceof_model)
    comps_phase = ceof_rotator.components_phase()

    assert isinstance(comps_phase, xr.DataArray)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_scores_amplitude(ceof_model):
    ceof_rotator = HilbertEOFRotator(n_modes=3)
    ceof_rotator.fit(ceof_model)
    scores_amp = ceof_rotator.scores_amplitude()

    assert isinstance(scores_amp, xr.DataArray)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_scores_phase(ceof_model):
    ceof_rotator = HilbertEOFRotator(n_modes=3)
    ceof_rotator.fit(ceof_model)
    scores_phase = ceof_rotator.scores_phase()

    assert isinstance(scores_phase, xr.DataArray)
