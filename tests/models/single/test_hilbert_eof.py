import warnings

import numpy as np
import pytest

from xeofs.single import HilbertEOF

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_fit(mock_data_array, dim):
    """Test fitting a HilbertEOF model"""
    # Create a xarray DataArray with random data
    ceof = HilbertEOF(n_modes=2)
    ceof.fit(mock_data_array, dim)

    # Check that the fit method has properly populated the attributes
    assert hasattr(ceof, "preprocessor")
    assert hasattr(ceof, "data")


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components_amplitude(mock_data_array, dim):
    """Test computation of components amplitude in HilbertEOF model"""
    ceof = HilbertEOF(n_modes=2)
    ceof.fit(mock_data_array, dim)

    comp_amp = ceof.components_amplitude()
    assert comp_amp is not None
    assert (comp_amp.fillna(0) >= 0).all()  # type: ignore


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components_phase(mock_data_array, dim):
    """Test computation of components phase in HilbertEOF model"""
    ceof = HilbertEOF(n_modes=2)
    ceof.fit(mock_data_array, dim)

    comp_phase = ceof.components_phase()
    assert comp_phase is not None
    assert ((-np.pi <= comp_phase.fillna(0)) & (comp_phase.fillna(0) <= np.pi)).all()  # type: ignore


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_scores_amplitude(mock_data_array, dim):
    """Test computation of scores amplitude in HilbertEOF model"""
    ceof = HilbertEOF(n_modes=2)
    ceof.fit(mock_data_array, dim)

    scores_amp = ceof.scores_amplitude()
    assert scores_amp is not None
    assert (scores_amp.fillna(0) >= 0).all()  # type: ignore


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_scores_phase(mock_data_array, dim):
    """Test computation of scores phase in HilbertEOF model"""
    ceof = HilbertEOF(n_modes=2)
    ceof.fit(mock_data_array, dim)

    scores_phase = ceof.scores_phase()
    assert scores_phase is not None
    assert (
        (-np.pi <= scores_phase.fillna(0)) & (scores_phase.fillna(0) <= np.pi)
    ).all()  # type: ignore


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_compute(mock_dask_data_array, dim):
    """Test computation of all attributes in HilbertEOF model"""
    ceof = HilbertEOF(n_modes=2)
    with pytest.raises(NotImplementedError):
        ceof.fit(mock_dask_data_array, dim)
