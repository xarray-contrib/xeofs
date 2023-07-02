import numpy as np
import xarray as xr
import pytest
from dask.array import Array as DaskArray  # type: ignore
from sklearn.utils.extmath import randomized_svd as svd
from scipy.sparse.linalg import svds as complex_svd  # type: ignore
from dask.array.linalg import svd_compressed as dask_svd
from xeofs.models.decomposer import Decomposer, CrossDecomposer


@pytest.fixture
def decomposer():
    return Decomposer(n_components=2, n_iter=3, random_state=42, verbose=False)

@pytest.fixture
def cross_decomposer():
    return CrossDecomposer(n_components=2, n_iter=3, random_state=42, verbose=False)

@pytest.fixture
def test_DataArray(test_DataArray):
    return test_DataArray.stack(sample=('time',), feature=('x', 'y')).dropna('feature')

@pytest.fixture
def test_DaskDataArray(test_DataArray):
    return test_DataArray.chunk({'sample': 1})

@pytest.fixture
def test_complex_DataArray(test_DataArray):
    return test_DataArray * (1 + 1j)

@pytest.fixture
def test_complex_DaskDataArray(test_complex_DataArray):
    return test_complex_DataArray.chunk({'sample': 1})


def test_decomposer_init(decomposer):
    assert decomposer.params['n_components'] == 2
    assert decomposer.params['n_iter'] == 3
    assert decomposer.params['random_state'] == 42
    assert decomposer.params['verbose'] == False

def test_cross_decomposer_init(cross_decomposer):
    assert cross_decomposer.params['n_components'] == 2
    assert cross_decomposer.params['n_iter'] == 3
    assert cross_decomposer.params['random_state'] == 42
    assert cross_decomposer.params['verbose'] == False

def test_decomposer_fit(decomposer, test_DataArray):
    decomposer.fit(test_DataArray)
    assert 'scores_' in decomposer.__dict__
    assert 'singular_values_' in decomposer.__dict__
    assert 'components_' in decomposer.__dict__
    
def test_decomposer_fit_dask(decomposer, test_DaskDataArray):
    decomposer.fit(test_DaskDataArray)
    assert 'scores_' in decomposer.__dict__
    assert 'singular_values_' in decomposer.__dict__
    assert 'components_' in decomposer.__dict__

def test_decomposer_fit_complex(decomposer, test_complex_DataArray):
    decomposer.fit(test_complex_DataArray)
    assert 'scores_' in decomposer.__dict__
    assert 'singular_values_' in decomposer.__dict__
    assert 'components_' in decomposer.__dict__

def test_cross_decomposer_fit(cross_decomposer, test_DataArray):
    cross_decomposer.fit(test_DataArray, test_DataArray)
    assert 'singular_vectors1_' in cross_decomposer.__dict__
    assert 'singular_values_' in cross_decomposer.__dict__
    assert 'singular_vectors2_' in cross_decomposer.__dict__

def test_cross_decomposer_fit_complex(cross_decomposer, test_complex_DataArray):
    cross_decomposer.fit(test_complex_DataArray, test_complex_DataArray)
    assert 'singular_vectors1_' in cross_decomposer.__dict__
    assert 'singular_values_' in cross_decomposer.__dict__
    assert 'singular_vectors2_' in cross_decomposer.__dict__

def test_cross_decomposer_fit_dask(cross_decomposer, test_DaskDataArray):
    cross_decomposer.fit(test_DaskDataArray, test_DaskDataArray)
    assert 'singular_vectors1_' in cross_decomposer.__dict__
    assert 'singular_values_' in cross_decomposer.__dict__
    assert 'singular_vectors2_' in cross_decomposer.__dict__

def test_cross_decomposer_fit_same_samples(cross_decomposer, test_DataArray):
    with pytest.raises(ValueError):
        cross_decomposer.fit(test_DataArray, test_DataArray.isel(sample=slice(1,3)))
