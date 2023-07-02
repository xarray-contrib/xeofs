import pytest
import numpy as np
import xarray as xr
from dask.array import Array as DaskArray   # type: ignore

from xeofs.models import EOF, ComplexEOF, EOFRotator, ComplexEOFRotator


@pytest.fixture
def eof_model(test_DataArray):
    eof = EOF(n_modes=5)
    eof.fit(test_DataArray, dim='time')
    return eof

@pytest.fixture
def eof_model_delayed(test_DaskDataArray):
    eof = EOF(n_modes=5)
    eof.fit(test_DaskDataArray, dim='time')
    return eof

@pytest.fixture
def ceof_model(test_DataArray):
    ceof = ComplexEOF(n_modes=5)
    ceof.fit(test_DataArray, dim='time')
    return ceof

@pytest.fixture
def ceof_model_delayed(test_DaskDataArray):
    ceof = ComplexEOF(n_modes=5)
    ceof.fit(test_DaskDataArray, dim='time')
    return ceof

def test_EOFRotator_init():
    # Instantiate the EOFRotator class
    eof_rotator = EOFRotator(3, power=2, max_iter=100, rtol=1e-6)

    assert eof_rotator._params['n_rot'] == 3
    assert eof_rotator._params['power'] == 2
    assert eof_rotator._params['max_iter'] == 100
    assert eof_rotator._params['rtol'] == 1e-6

def test_EOFRotator_fit(eof_model):
    eof_rotator = EOFRotator(3)
    eof_rotator.fit(eof_model)

    assert hasattr(eof_rotator, '_model')
    assert hasattr(eof_rotator, '_rotation_matrix')
    assert hasattr(eof_rotator, '_idx_expvar')
    assert hasattr(eof_rotator, '_explained_variance')
    assert hasattr(eof_rotator, '_explained_variance_ratio')
    assert hasattr(eof_rotator, '_components')
    assert hasattr(eof_rotator, '_scores')
    
def test_EOFRotator_transform(eof_model, test_DataArray):
    eof_rotator = EOFRotator(3)
    eof_rotator.fit(eof_model)
    projections = eof_rotator.transform(test_DataArray)
    
    assert isinstance(projections, xr.DataArray)
    
def test_EOFRotator_inverse_transform(eof_model):
    eof_rotator = EOFRotator(3)
    eof_rotator.fit(eof_model)
    Xrec = eof_rotator.inverse_transform()

    assert isinstance(Xrec, xr.DataArray)

def test_EOFRotator_explained_variance(eof_model):
    eof_rotator = EOFRotator(3)
    eof_rotator.fit(eof_model)
    exp_var = eof_rotator.explained_variance()

    assert isinstance(exp_var, xr.DataArray)

def test_EOFRotator_explained_variance_ratio(eof_model):
    eof_rotator = EOFRotator(3)
    eof_rotator.fit(eof_model)
    exp_var_ratio = eof_rotator.explained_variance_ratio()

    assert isinstance(exp_var_ratio, xr.DataArray)

def test_EOFRotator_components(eof_model):
    eof_rotator = EOFRotator(3)
    eof_rotator.fit(eof_model)
    components = eof_rotator.components()

    assert isinstance(components, xr.DataArray)

def test_EOFRotator_scores(eof_model):
    eof_rotator = EOFRotator(3)
    eof_rotator.fit(eof_model)
    scores = eof_rotator.scores()

    assert isinstance(scores, xr.DataArray)

def test_EOFRotator_compute(eof_model_delayed):
    eof_rotator = EOFRotator(5)
    eof_rotator.fit(eof_model_delayed)
    
    # before computation, the attributes should be dask arrays
    assert isinstance(eof_rotator._explained_variance.data, DaskArray), "The attribute _explained_variance should be a dask array."
    assert isinstance(eof_rotator._explained_variance_ratio.data, DaskArray)
    assert isinstance(eof_rotator._components.data, DaskArray)
    assert isinstance(eof_rotator._rotation_matrix.data, DaskArray)
    assert isinstance(eof_rotator._scores.data, DaskArray)

    eof_rotator.compute()

    # after computation, the attributes should be numpy ndarrays
    assert isinstance(eof_rotator._explained_variance.data, np.ndarray)
    assert isinstance(eof_rotator._explained_variance_ratio.data, np.ndarray)
    assert isinstance(eof_rotator._components.data, np.ndarray)
    assert isinstance(eof_rotator._rotation_matrix.data, np.ndarray)
    assert isinstance(eof_rotator._scores.data, np.ndarray)

def test_ComplexEOFRotator_init():
    # Instantiate the ComplexEOFRotator class
    ceof_rotator = ComplexEOFRotator(3, power=2, max_iter=100, rtol=1e-6)

    assert ceof_rotator._params['n_rot'] == 3
    assert ceof_rotator._params['power'] == 2
    assert ceof_rotator._params['max_iter'] == 100
    assert ceof_rotator._params['rtol'] == 1e-6

def test_ComplexEOFRotator_fit(ceof_model):
    ceof_rotator = ComplexEOFRotator(3)
    ceof_rotator.fit(ceof_model)

    assert hasattr(ceof_rotator, '_model')
    assert hasattr(ceof_rotator, '_rotation_matrix')
    assert hasattr(ceof_rotator, '_idx_expvar')
    assert hasattr(ceof_rotator, '_explained_variance')
    assert hasattr(ceof_rotator, '_explained_variance_ratio')
    assert hasattr(ceof_rotator, '_components')
    assert hasattr(ceof_rotator, '_scores')

def test_ComplexEOFRotator_transform_not_implemented(ceof_model, test_DataArray):
    ceof_rotator = ComplexEOFRotator(3)
    ceof_rotator.fit(ceof_model)

    with pytest.raises(NotImplementedError):
        ceof_rotator.transform(test_DataArray)
    
def test_ComplexEOFRotator_inverse_transform(ceof_model):
    ceof_rotator = ComplexEOFRotator(3)
    ceof_rotator.fit(ceof_model)
    Xrec = ceof_rotator.inverse_transform()

    assert isinstance(Xrec, xr.DataArray)

def test_ComplexEOFRotator_components_amplitude(ceof_model):
    ceof_rotator = ComplexEOFRotator(3)
    ceof_rotator.fit(ceof_model)
    comps_amp = ceof_rotator.components_amplitude()

    assert isinstance(comps_amp, xr.DataArray)

def test_ComplexEOFRotator_components_phase(ceof_model):
    ceof_rotator = ComplexEOFRotator(3)
    ceof_rotator.fit(ceof_model)
    comps_phase = ceof_rotator.components_phase()

    assert isinstance(comps_phase, xr.DataArray)

def test_ComplexEOFRotator_scores_amplitude(ceof_model):
    ceof_rotator = ComplexEOFRotator(3)
    ceof_rotator.fit(ceof_model)
    scores_amp = ceof_rotator.scores_amplitude()

    assert isinstance(scores_amp, xr.DataArray)

def test_ComplexEOFRotator_scores_phase(ceof_model):
    ceof_rotator = ComplexEOFRotator(3)
    ceof_rotator.fit(ceof_model)
    scores_phase = ceof_rotator.scores_phase()

    assert isinstance(scores_phase, xr.DataArray)
