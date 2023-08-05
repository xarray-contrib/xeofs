import pytest
import numpy as np
import xarray as xr
from dask.array import Array as DaskArray  # type: ignore

# Import the classes from your modules
from xeofs.models import MCA, MCARotator


@pytest.fixture
def mca_model(mock_data_array, dim):
    mca = MCA(n_modes=5)
    mca.fit(mock_data_array, mock_data_array, dim)
    return mca


@pytest.fixture
def mca_model_delayed(mock_dask_data_array, dim):
    mca = MCA(n_modes=5)
    mca.fit(mock_dask_data_array, mock_dask_data_array, dim)
    return mca


def test_init():
    mca_rotator = MCARotator(n_modes=4)
    assert mca_rotator._params["n_modes"] == 4
    assert mca_rotator._params["power"] == 1
    assert mca_rotator._params["max_iter"] == 1000
    assert mca_rotator._params["rtol"] == 1e-8
    assert mca_rotator._params["squared_loadings"] == False


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_fit(mca_model):
    mca_rotator = MCARotator(n_modes=4)
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
    mca_rotator = MCARotator(n_modes=4)
    mca_rotator.fit(mca_model)

    projections = mca_rotator.transform(data1=mock_data_array, data2=mock_data_array)

    assert len(projections) == 2


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_inverse_transform(mca_model):
    mca_rotator = MCARotator(n_modes=4)
    mca_rotator.fit(mca_model)

    reconstructed_data = mca_rotator.inverse_transform(mode=slice(1, 3))

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
    mca_rotator = MCARotator(n_modes=4)
    mca_rotator.fit(mca_model)
    covariance_fraction = mca_rotator.squared_covariance()
    assert isinstance(covariance_fraction, xr.DataArray)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_squared_covariance_fraction(mca_model, mock_data_array, dim):
    mca_rotator = MCARotator(n_modes=4)
    mca_rotator.fit(mca_model)
    squared_covariance_fraction = mca_rotator.squared_covariance_fraction()
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
    mca_rotator = MCARotator(n_modes=4)
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
    mca_rotator = MCARotator(n_modes=4)
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
    mca_rotator = MCARotator(n_modes=4)
    mca_rotator.fit(mca_model)
    comps1, comps2 = mca_rotator.components()
    assert isinstance(comps1, xr.DataArray)
    assert isinstance(comps2, xr.DataArray)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_scores(mca_model, mock_data_array, dim):
    mca_rotator = MCARotator(n_modes=4)
    mca_rotator.fit(mca_model)
    scores1, scores2 = mca_rotator.scores()
    assert isinstance(scores1, xr.DataArray)
    assert isinstance(scores2, xr.DataArray)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_homogeneous_patterns(mca_model, mock_data_array, dim):
    mca_rotator = MCARotator(n_modes=4)
    mca_rotator.fit(mca_model)
    patterns, pvalues = mca_rotator.homogeneous_patterns()
    assert isinstance(patterns[0], xr.DataArray)
    assert isinstance(patterns[1], xr.DataArray)
    assert isinstance(pvalues[0], xr.DataArray)
    assert isinstance(pvalues[1], xr.DataArray)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_heterogeneous_patterns(mca_model, mock_data_array, dim):
    mca_rotator = MCARotator(n_modes=4)
    mca_rotator.fit(mca_model)
    patterns, pvalues = mca_rotator.heterogeneous_patterns()
    assert isinstance(patterns[0], xr.DataArray)
    assert isinstance(patterns[1], xr.DataArray)
    assert isinstance(pvalues[0], xr.DataArray)
    assert isinstance(pvalues[1], xr.DataArray)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_compute(mca_model_delayed):
    """Test the compute method of the MCARotator class."""

    mca_rotator = MCARotator(n_modes=4, rtol=1e-5)
    mca_rotator.fit(mca_model_delayed)

    assert isinstance(
        mca_rotator.data.squared_covariance.data, DaskArray
    ), "squared_covariance is not a delayed object"
    assert isinstance(
        mca_rotator.data.components1.data, DaskArray
    ), "components1 is not a delayed object"
    assert isinstance(
        mca_rotator.data.components2.data, DaskArray
    ), "components2 is not a delayed object"
    assert isinstance(
        mca_rotator.data.scores1.data, DaskArray
    ), "scores1 is not a delayed object"
    assert isinstance(
        mca_rotator.data.scores2.data, DaskArray
    ), "scores2 is not a delayed object"
    assert isinstance(
        mca_rotator.data.rotation_matrix.data, DaskArray
    ), "rotation_matrix is not a delayed object"
    assert isinstance(
        mca_rotator.data.phi_matrix.data, DaskArray
    ), "phi_matrix is not a delayed object"
    assert isinstance(
        mca_rotator.data.norm1.data, DaskArray
    ), "norm1 is not a delayed object"
    assert isinstance(
        mca_rotator.data.norm2.data, DaskArray
    ), "norm2 is not a delayed object"
    assert isinstance(
        mca_rotator.data.modes_sign.data, DaskArray
    ), "modes_sign is not a delayed object"

    mca_rotator.compute()

    assert isinstance(
        mca_rotator.data.squared_covariance.data, np.ndarray
    ), "squared_covariance is not computed"
    assert isinstance(
        mca_rotator.data.total_squared_covariance.data, np.ndarray
    ), "total_squared_covariance is not computed"
    assert isinstance(
        mca_rotator.data.components1.data, np.ndarray
    ), "components1 is not computed"
    assert isinstance(
        mca_rotator.data.components2.data, np.ndarray
    ), "components2 is not computed"
    assert isinstance(
        mca_rotator.data.scores1.data, np.ndarray
    ), "scores1 is not computed"
    assert isinstance(
        mca_rotator.data.scores2.data, np.ndarray
    ), "scores2 is not computed"
    assert isinstance(
        mca_rotator.data.rotation_matrix.data, np.ndarray
    ), "rotation_matrix is not computed"
    assert isinstance(
        mca_rotator.data.phi_matrix.data, np.ndarray
    ), "phi_matrix is not computed"
    assert isinstance(mca_rotator.data.norm1.data, np.ndarray), "norm1 is not computed"
    assert isinstance(mca_rotator.data.norm2.data, np.ndarray), "norm2 is not computed"
    assert isinstance(
        mca_rotator.data.modes_sign.data, np.ndarray
    ), "modes_sign is not computed"
