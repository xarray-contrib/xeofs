import pytest
import numpy as np
import xarray as xr
import dask.array as da
from numpy.testing import assert_allclose

from xeofs.models.complex_mca import ComplexMCA


def test_complex_mca_initialization():
    mca = ComplexMCA()
    assert mca is not None


def test_complex_mca_fit(test_DataArray):
    mca = ComplexMCA()
    mca.fit(test_DataArray, test_DataArray, dims='time')
    assert mca._singular_values is not None
    assert mca._explained_variance is not None
    assert mca._squared_total_variance is not None
    assert mca._singular_vectors1 is not None
    assert mca._singular_vectors2 is not None
    assert mca._norm1 is not None
    assert mca._norm2 is not None

def test_complex_mca_fit_empty_data():
    mca = ComplexMCA()
    with pytest.raises(ValueError):
        mca.fit(xr.DataArray(), xr.DataArray(), dims='time')


def test_complex_mca_fit_invalid_dims(test_DataArray):
    mca = ComplexMCA()
    with pytest.raises(ValueError):
        mca.fit(test_DataArray, test_DataArray, dims=('invalid_dim1', 'invalid_dim2'))


def test_complex_mca_transform_not_implemented(test_DataArray):
    mca = ComplexMCA()
    with pytest.raises(NotImplementedError):
        mca.transform(test_DataArray, test_DataArray)


def test_complex_mca_homogeneous_patterns_not_implemented(test_DataArray):
    mca = ComplexMCA()
    with pytest.raises(NotImplementedError):
        mca.homogeneous_patterns()


def test_complex_mca_heterogeneous_patterns_not_implemented(test_DataArray):
    mca = ComplexMCA()
    with pytest.raises(NotImplementedError):
        mca.heterogeneous_patterns()

def test_complex_mca_fit_with_dataset(test_Dataset):
    mca = ComplexMCA()
    mca.fit(test_Dataset, test_Dataset, dims='time')
    assert mca._singular_values is not None


def test_complex_mca_fit_with_dataarraylist(test_DataArrayList):
    mca = ComplexMCA()
    mca.fit(test_DataArrayList, test_DataArrayList, dims='time')
    assert mca._singular_values is not None