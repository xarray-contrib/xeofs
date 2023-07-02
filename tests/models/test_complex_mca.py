import pytest
import numpy as np
import xarray as xr
import dask.array as da
from numpy.testing import assert_allclose

from xeofs.models import ComplexMCA

@pytest.fixture
def mca_model():
    return ComplexMCA(n_modes=3)

def test_complex_mca_initialization():
    mca = ComplexMCA(n_modes=1)
    assert mca is not None


def test_complex_mca_fit(mca_model, test_DataArray):
    mca_model.fit(test_DataArray, test_DataArray, dims='time')
    assert mca_model._singular_values is not None
    assert mca_model._explained_variance is not None
    assert mca_model._squared_total_variance is not None
    assert mca_model._singular_vectors1 is not None
    assert mca_model._singular_vectors2 is not None
    assert mca_model._norm1 is not None
    assert mca_model._norm2 is not None

def test_complex_mca_fit_empty_data():
    mca = ComplexMCA()
    with pytest.raises(ValueError):
        mca.fit(xr.DataArray(), xr.DataArray(), dims='time')


def test_complex_mca_fit_invalid_dims(mca_model, test_DataArray):
    with pytest.raises(ValueError):
        mca_model.fit(test_DataArray, test_DataArray, dims=('invalid_dim1', 'invalid_dim2'))


def test_complex_mca_transform_not_implemented(mca_model, test_DataArray):
    with pytest.raises(NotImplementedError):
        mca_model.transform(test_DataArray, test_DataArray)


def test_complex_mca_homogeneous_patterns_not_implemented():
    mca = ComplexMCA()
    with pytest.raises(NotImplementedError):
        mca.homogeneous_patterns()


def test_complex_mca_heterogeneous_patterns_not_implemented():
    mca = ComplexMCA()
    with pytest.raises(NotImplementedError):
        mca.heterogeneous_patterns()

def test_complex_mca_fit_with_dataset(mca_model, test_Dataset):
    mca_model.fit(test_Dataset, test_Dataset, dims='time')
    assert mca_model._singular_values is not None


def test_complex_mca_fit_with_dataarraylist(mca_model, test_DataArrayList):
    mca_model.fit(test_DataArrayList, test_DataArrayList, dims='time')
    assert mca_model._singular_values is not None