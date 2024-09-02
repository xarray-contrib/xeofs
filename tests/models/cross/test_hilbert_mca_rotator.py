import numpy as np
import pytest
import xarray as xr

# Import the classes from your modules
from xeofs.cross import HilbertMCA, HilbertMCARotator


@pytest.fixture
def mca_model(mock_data_array, dim):
    mca = HilbertMCA(n_modes=5)
    mca.fit(mock_data_array, mock_data_array, dim)
    return mca


@pytest.fixture
def mca_model_delayed(mock_dask_data_array, dim):
    mca = HilbertMCA(n_modes=5)
    mca.fit(mock_dask_data_array, mock_dask_data_array, dim)
    return mca


def test_init():
    mca_rotator = HilbertMCARotator(n_modes=2)
    assert mca_rotator._params["n_modes"] == 2
    assert mca_rotator._params["power"] == 1
    assert mca_rotator._params["max_iter"] == 1000
    assert mca_rotator._params["rtol"] == 1e-8


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_fit(mca_model):
    mca_rotator = HilbertMCARotator(n_modes=2)
    mca_rotator.fit(mca_model)

    assert hasattr(mca_rotator, "model")
    assert hasattr(mca_rotator, "data")


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_transform(mca_model, mock_data_array):
    mca_rotator = HilbertMCARotator(n_modes=2)
    mca_rotator.fit(mca_model)

    with pytest.raises(NotImplementedError):
        mca_rotator.transform(X=mock_data_array, Y=mock_data_array)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_inverse_transform(mca_model):
    mca_rotator = HilbertMCARotator(n_modes=2)
    mca_rotator.fit(mca_model)

    scores1 = mca_rotator.data["scores1"].sel(mode=slice(1, 3))
    scores2 = mca_rotator.data["scores2"].sel(mode=slice(1, 3))

    reconstructed_data = mca_rotator.inverse_transform(scores1, scores2)

    assert isinstance(reconstructed_data, list)
    assert len(reconstructed_data) == 2


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_squared_covariance_fraction(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    squared_covariance_fraction = mca_model.squared_covariance_fraction()
    assert isinstance(squared_covariance_fraction, xr.DataArray)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_covariance_fraction(mca_model):
    mca_rotator = HilbertMCARotator(n_modes=4)
    mca_rotator.fit(mca_model)
    cf = mca_rotator.covariance_fraction_CD95()
    assert isinstance(cf, xr.DataArray)
    assert all(cf <= 1), "Squared covariance fraction is greater than 1"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components(mca_model, mock_data_array, dim):
    mca_rotator = HilbertMCARotator(n_modes=2)
    mca_rotator.fit(mca_model)
    comps1, comps2 = mca_rotator.components()
    assert isinstance(comps1, xr.DataArray)
    assert isinstance(comps2, xr.DataArray)
    # assert that the components are Hilbert valued
    assert np.iscomplexobj(comps1)
    assert np.iscomplexobj(comps2)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_scores(mca_model, mock_data_array, dim):
    mca_rotator = HilbertMCARotator(n_modes=2)
    mca_rotator.fit(mca_model)
    scores1, scores2 = mca_rotator.scores()
    assert isinstance(scores1, xr.DataArray)
    assert isinstance(scores2, xr.DataArray)
    # assert that the scores are Hilbert valued
    assert np.iscomplexobj(scores1)
    assert np.iscomplexobj(scores2)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_homogeneous_patterns(mca_model, mock_data_array, dim):
    mca_rotator = HilbertMCARotator(n_modes=2)
    mca_rotator.fit(mca_model)
    mca_rotator.homogeneous_patterns()


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_heterogeneous_patterns(mca_model, mock_data_array, dim):
    mca_rotator = HilbertMCARotator(n_modes=2)
    mca_rotator.fit(mca_model)
    mca_rotator.heterogeneous_patterns()


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components_amplitude(mca_model, mock_data_array, dim):
    mca_rotator = HilbertMCARotator(n_modes=2)
    mca_rotator.fit(mca_model)
    amps1, amps2 = mca_rotator.components_amplitude()


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components_phase(mca_model, mock_data_array, dim):
    mca_rotator = HilbertMCARotator(n_modes=2)
    mca_rotator.fit(mca_model)
    amps1, amps2 = mca_rotator.components_phase()


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_scores_amplitude(mca_model, mock_data_array, dim):
    mca_rotator = HilbertMCARotator(n_modes=2)
    mca_rotator.fit(mca_model)
    amps1, amps2 = mca_rotator.scores_amplitude()


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_scores_phase(mca_model, mock_data_array, dim):
    mca_rotator = HilbertMCARotator(n_modes=2)
    mca_rotator.fit(mca_model)
    amps1, amps2 = mca_rotator.scores_phase()
