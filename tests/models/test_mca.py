import pytest
import numpy as np
import xarray as xr
import dask.array as da
from numpy.testing import assert_allclose

from xeofs.models.mca import MCA


def test_mca_initialization():
    mca = MCA()
    assert mca is not None


def test_mca_fit(test_DataArray):
    mca = MCA()
    mca.fit(test_DataArray, test_DataArray, dims='time')
    assert mca._singular_values is not None
    assert mca._explained_variance is not None
    assert mca._squared_total_variance is not None
    assert mca._singular_vectors1 is not None
    assert mca._singular_vectors2 is not None
    assert mca._norm1 is not None
    assert mca._norm2 is not None


def test_mca_fit_empty_data():
    mca = MCA()
    with pytest.raises(ValueError):
        mca.fit(xr.DataArray(), xr.DataArray(), dims='time')

def test_mca_fit_invalid_dims(test_DataArray):
    mca = MCA()
    with pytest.raises(ValueError):
        mca.fit(test_DataArray, test_DataArray, dims=('invalid_dim1', 'invalid_dim2'))


def test_mca_fit_with_dataset(test_Dataset):
    mca = MCA()
    mca.fit(test_Dataset, test_Dataset, dims='time')
    assert mca._singular_values is not None


def test_mca_fit_with_dataarraylist(test_DataArrayList):
    mca = MCA()
    mca.fit(test_DataArrayList, test_DataArrayList, dims='time')
    assert mca._singular_values is not None







