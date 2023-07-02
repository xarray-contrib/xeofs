import numpy as np
import xarray as xr
import pytest
import dask.array as da
import warnings
from numpy.testing import assert_allclose

from xeofs.models import EOF, ComplexEOF

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")




def test_ComplexEOF_fit(test_DataArray):
    """Test fitting a ComplexEOF model"""
    # Create a xarray DataArray with random data
    dims = 'time'
    
    ceof = ComplexEOF(n_modes=2)
    ceof.fit(test_DataArray, dims)

    # Check that the fit method has properly populated the attributes
    assert ceof._total_variance is not None
    assert ceof._singular_values is not None
    assert ceof._explained_variance is not None
    assert ceof._explained_variance_ratio is not None
    assert ceof._components is not None
    assert ceof._scores is not None


def test_ComplexEOF_components_amplitude(test_DataArray):
    """Test computation of components amplitude in ComplexEOF model"""
    dims = 'time'
    ceof = ComplexEOF(n_modes=2)
    ceof.fit(test_DataArray, dims)

    comp_amp = ceof.components_amplitude()
    assert comp_amp is not None
    assert (comp_amp.fillna(0) >= 0).all()  #type: ignore


def test_ComplexEOF_components_phase(test_DataArray):
    """Test computation of components phase in ComplexEOF model"""
    dims = 'time'
    ceof = ComplexEOF(n_modes=2)
    ceof.fit(test_DataArray, dims)

    comp_phase = ceof.components_phase()
    assert comp_phase is not None
    assert ((-np.pi <= comp_phase.fillna(0)) & (comp_phase.fillna(0) <= np.pi)).all()  #type: ignore


def test_ComplexEOF_scores_amplitude(test_DataArray):
    """Test computation of scores amplitude in ComplexEOF model"""
    dims = 'time'
    ceof = ComplexEOF(n_modes=2)
    ceof.fit(test_DataArray, dims)

    scores_amp = ceof.scores_amplitude()
    assert scores_amp is not None
    assert (scores_amp.fillna(0) >= 0).all()  #type: ignore


def test_ComplexEOF_scores_phase(test_DataArray):
    """Test computation of scores phase in ComplexEOF model"""
    dims = 'time'
    ceof = ComplexEOF(n_modes=2)
    ceof.fit(test_DataArray, dims)

    scores_phase = ceof.scores_phase()
    assert scores_phase is not None
    assert ((-np.pi <= scores_phase.fillna(0)) & (scores_phase.fillna(0) <= np.pi)).all()  #type: ignore


def test_ComplexEOF_compute(test_DaskDataArray):
    """Test computation of all attributes in ComplexEOF model"""
    dims = 'time'
    ceof = ComplexEOF(n_modes=2)
    with pytest.raises(NotImplementedError):
        ceof.fit(test_DaskDataArray, dims)
