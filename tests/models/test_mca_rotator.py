import pytest
import numpy as np
import xarray as xr
from dask.array import Array as DaskArray  # type: ignore

# Import the classes from your modules
from xeofs.models import MCA, MCARotator
from ..utilities import data_is_dask


@pytest.fixture
def mca_model(mock_data_array, dim):
    mca = MCA(n_modes=5)
    mca.fit(mock_data_array, mock_data_array, dim)
    return mca


@pytest.fixture
def mca_model_delayed(mock_dask_data_array, dim):
    mca = MCA(n_modes=5, compute=False)
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
def test_compute(mca_model_delayed, compute):
    """Test the compute method of the MCARotator class."""

    mca_rotator = MCARotator(n_modes=4, compute=compute, rtol=1e-5)
    mca_rotator.fit(mca_model_delayed)

    if compute:
        assert not data_is_dask(mca_rotator.data["squared_covariance"])
        assert not data_is_dask(mca_rotator.data["components1"])
        assert not data_is_dask(mca_rotator.data["components2"])
        assert not data_is_dask(mca_rotator.data["rotation_matrix"])
        assert not data_is_dask(mca_rotator.data["phi_matrix"])
        assert not data_is_dask(mca_rotator.data["norm1"])
        assert not data_is_dask(mca_rotator.data["norm2"])
        assert not data_is_dask(mca_rotator.data["modes_sign"])

    else:
        assert data_is_dask(mca_rotator.data["squared_covariance"])
        assert data_is_dask(mca_rotator.data["components1"])
        assert data_is_dask(mca_rotator.data["components2"])
        assert data_is_dask(mca_rotator.data["rotation_matrix"])
        assert data_is_dask(mca_rotator.data["phi_matrix"])
        assert data_is_dask(mca_rotator.data["norm1"])
        assert data_is_dask(mca_rotator.data["norm2"])
        assert data_is_dask(mca_rotator.data["modes_sign"])


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_save_load(dim, mock_data_array, tmp_path):
    """Test save/load methods in MCA class, ensuring that we can
    roundtrip the model and get the same results when transforming
    data."""
    original_unrotated = MCA()
    original_unrotated.fit(mock_data_array, mock_data_array, dim)

    original = MCARotator()
    original.fit(original_unrotated)

    # Save the EOF model
    original.save(tmp_path / "mca.zarr")

    # Check that the EOF model has been saved
    assert (tmp_path / "mca.zarr").exists()

    # Recreate the model from saved file
    loaded = MCARotator.load(tmp_path / "mca.zarr")

    # Check that the params and DataContainer objects match
    assert original.get_params() == loaded.get_params()
    assert all([key in loaded.data for key in original.data])
    assert all(
        [
            loaded.data._allow_compute[key] == original.data._allow_compute[key]
            for key in original.data
        ]
    )

    # Test that the recreated model can be used to transform new data
    assert np.allclose(
        original.scores(),
        loaded.transform(data1=mock_data_array, data2=mock_data_array),
        rtol=1e-3,
        atol=1e-3,
    )

    # Enhancement: the loaded model should also be able to inverse_transform new data
    # assert np.allclose(
    #     original.inverse_transform(original.scores()),
    #     loaded.inverse_transform(loaded.scores()),
    #     rtol=1e-3,
    #     atol=1e-3,
    # )
