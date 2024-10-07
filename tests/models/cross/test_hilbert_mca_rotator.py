import numpy as np
import pytest
import xarray as xr

# Import the classes from your modules
from xeofs.cross import HilbertMCA, HilbertMCARotator

from ...utilities import skip_if_missing_engine


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

    assert hasattr(mca_rotator, "model_data")
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


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
# Currently, netCDF4 does not support complex numbers, so skip this test
@pytest.mark.parametrize("engine", ["h5netcdf", "zarr"])
def test_save_load_with_data(tmp_path, engine, mca_model):
    """Test save/load methods in HilbertMCARotator class, ensuring that we can
    roundtrip the model and get the same results."""
    skip_if_missing_engine(engine)

    original = HilbertMCARotator(n_modes=2)
    original.fit(mca_model)

    # Save the HilbertMCARotator model
    original.save(tmp_path / "mca", engine=engine, save_data=True)

    # Check that the HilbertMCARotator model has been saved
    assert (tmp_path / "mca").exists()

    # Recreate the model from saved file
    loaded = HilbertMCARotator.load(tmp_path / "mca", engine=engine)

    # Check that the params and DataContainer objects match
    assert original.get_params() == loaded.get_params()
    assert all([key in loaded.data for key in original.data])
    for key in original.data:
        assert loaded.data[key].equals(original.data[key])

    # Test that the recreated model can compute the SCF
    assert np.allclose(
        original.squared_covariance_fraction(), loaded.squared_covariance_fraction()
    )

    # Test that the recreated model can compute the components amplitude
    A1_original, A2_original = original.components_amplitude()
    A1_loaded, A2_loaded = loaded.components_amplitude()
    assert np.allclose(A1_original, A1_loaded)
    assert np.allclose(A2_original, A2_loaded)

    # Test that the recreated model can compute the components phase
    P1_original, P2_original = original.components_phase()
    P1_loaded, P2_loaded = loaded.components_phase()
    assert np.allclose(P1_original, P1_loaded)
    assert np.allclose(P2_original, P2_loaded)
