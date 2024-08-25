import pytest
import numpy as np
import xarray as xr

# Import the classes from your modules
from xeofs.models import ComplexMCA, ComplexMCARotator


@pytest.fixture
def mca_model(mock_data_array, dim):
    mca = ComplexMCA(n_modes=5)
    mca.fit(mock_data_array, mock_data_array, dim)
    return mca


@pytest.fixture
def mca_model_delayed(mock_dask_data_array, dim):
    mca = ComplexMCA(n_modes=5)
    mca.fit(mock_dask_data_array, mock_dask_data_array, dim)
    return mca


def test_init():
    mca_rotator = ComplexMCARotator(n_modes=2)
    assert mca_rotator._params["n_modes"] == 2
    assert mca_rotator._params["power"] == 1
    assert mca_rotator._params["max_iter"] == 1000
    assert mca_rotator._params["rtol"] == 1e-8
    assert mca_rotator._params["squared_loadings"] is False


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_fit(mca_model):
    mca_rotator = ComplexMCARotator(n_modes=2)
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
    mca_rotator = ComplexMCARotator(n_modes=2)
    mca_rotator.fit(mca_model)

    with pytest.raises(NotImplementedError):
        mca_rotator.transform(data1=mock_data_array, data2=mock_data_array)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_inverse_transform(mca_model):
    mca_rotator = ComplexMCARotator(n_modes=2)
    mca_rotator.fit(mca_model)

    scores1 = mca_rotator.data["scores1"].sel(mode=slice(1, 3))
    scores2 = mca_rotator.data["scores2"].sel(mode=slice(1, 3))

    reconstructed_data = mca_rotator.inverse_transform(scores1, scores2)

    assert isinstance(reconstructed_data, tuple)
    assert len(reconstructed_data) == 2


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_squared_covariance(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    squared_covariance = mca_model.squared_covariance()
    assert isinstance(squared_covariance, xr.DataArray)


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
def test_singular_values(mca_model):
    mca_rotator = ComplexMCARotator(n_modes=4)
    mca_rotator.fit(mca_model)
    n_modes = mca_rotator.get_params()["n_modes"]
    svals = mca_rotator.singular_values()
    assert isinstance(svals, xr.DataArray)
    assert svals.size == n_modes


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_covariance_fraction(mca_model):
    mca_rotator = ComplexMCARotator(n_modes=4)
    mca_rotator.fit(mca_model)
    cf = mca_rotator.covariance_fraction()
    assert isinstance(cf, xr.DataArray)
    assert cf.sum("mode") <= 1.00001, "Covariance fraction is greater than 1"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components(mca_model, mock_data_array, dim):
    mca_rotator = ComplexMCARotator(n_modes=2)
    mca_rotator.fit(mca_model)
    comps1, comps2 = mca_rotator.components()
    assert isinstance(comps1, xr.DataArray)
    assert isinstance(comps2, xr.DataArray)
    # assert that the components are complex valued
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
    mca_rotator = ComplexMCARotator(n_modes=2)
    mca_rotator.fit(mca_model)
    scores1, scores2 = mca_rotator.scores()
    assert isinstance(scores1, xr.DataArray)
    assert isinstance(scores2, xr.DataArray)
    # assert that the scores are complex valued
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
    mca_rotator = ComplexMCARotator(n_modes=2)
    mca_rotator.fit(mca_model)
    with pytest.raises(NotImplementedError):
        _ = mca_rotator.homogeneous_patterns()


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_heterogeneous_patterns(mca_model, mock_data_array, dim):
    mca_rotator = ComplexMCARotator(n_modes=2)
    mca_rotator.fit(mca_model)
    with pytest.raises(NotImplementedError):
        _ = mca_rotator.heterogeneous_patterns()


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components_amplitude(mca_model, mock_data_array, dim):
    mca_rotator = ComplexMCARotator(n_modes=2)
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
    mca_rotator = ComplexMCARotator(n_modes=2)
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
    mca_rotator = ComplexMCARotator(n_modes=2)
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
    mca_rotator = ComplexMCARotator(n_modes=2)
    mca_rotator.fit(mca_model)
    amps1, amps2 = mca_rotator.scores_phase()
