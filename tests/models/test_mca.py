import pytest
import numpy as np
import xarray as xr
import dask.array as da
from numpy.testing import assert_allclose

from xeofs.models.mca import MCA

@pytest.fixture
def mca_model():
    return MCA()

def test_mca_initialization():
    mca = MCA()
    assert mca is not None


def test_mca_fit(mca_model, test_DataArray):
    mca_model.fit(test_DataArray, test_DataArray, dims='time')
    assert mca_model._singular_values is not None
    assert mca_model._explained_variance is not None
    assert mca_model._squared_total_variance is not None
    assert mca_model._singular_vectors1 is not None
    assert mca_model._singular_vectors2 is not None
    assert mca_model._norm1 is not None
    assert mca_model._norm2 is not None


def test_mca_fit_empty_data(mca_model):
    with pytest.raises(ValueError):
        mca_model.fit(xr.DataArray(), xr.DataArray(), dims='time')

def test_mca_fit_invalid_dims(mca_model, test_DataArray):
    with pytest.raises(ValueError):
        mca_model.fit(test_DataArray, test_DataArray, dims=('invalid_dim1', 'invalid_dim2'))


def test_mca_fit_with_dataset(mca_model, test_Dataset):
    mca_model.fit(test_Dataset, test_Dataset, dims='time')
    assert mca_model._singular_values is not None


def test_mca_fit_with_dataarraylist(mca_model, test_DataArrayList):
    mca_model.fit(test_DataArrayList, test_DataArrayList, dims='time')
    assert mca_model._singular_values is not None


def test_transform(mca_model, test_DataArray):
    mca_model.fit(test_DataArray, test_DataArray, dims='time')
    result = mca_model.transform(data1=test_DataArray, data2=test_DataArray)
    assert isinstance(result, list)
    assert isinstance(result[0], xr.DataArray)
    
def test_inverse_transform(mca_model, test_DataArray):
    mca_model.fit(test_DataArray, test_DataArray, dims='time')
    # Assuming mode as 1 for simplicity
    Xrec1, Xrec2 = mca_model.inverse_transform(1)
    assert isinstance(Xrec1, xr.DataArray)
    assert isinstance(Xrec2, xr.DataArray)
    
def test_singular_values(mca_model, test_DataArray):
    mca_model.fit(test_DataArray, test_DataArray, dims='time')
    singular_values = mca_model.singular_values()
    assert isinstance(singular_values, xr.DataArray)

def test_explained_variance(mca_model, test_DataArray):
    mca_model.fit(test_DataArray, test_DataArray, dims='time')
    explained_variance = mca_model.explained_variance()
    assert isinstance(explained_variance, xr.DataArray)

def test_squared_covariance_fraction(mca_model, test_DataArray):
    mca_model.fit(test_DataArray, test_DataArray, dims='time')
    squared_covariance_fraction = mca_model.squared_covariance_fraction()
    assert isinstance(squared_covariance_fraction, xr.DataArray)

def test_components(mca_model, test_DataArray):
    mca_model.fit(test_DataArray, test_DataArray, dims='time')
    components1, components2 = mca_model.components()
    assert isinstance(components1, xr.DataArray)
    assert isinstance(components2, xr.DataArray)

def test_scores(mca_model, test_DataArray):
    mca_model.fit(test_DataArray, test_DataArray, dims='time')
    scores1, scores2 = mca_model.scores()
    assert isinstance(scores1, xr.DataArray)
    assert isinstance(scores2, xr.DataArray)

def test_homogeneous_patterns(mca_model, test_DataArray):
    mca_model.fit(test_DataArray, test_DataArray, dims='time')
    patterns1, patterns2, pvals1, pvals2 = mca_model.homogeneous_patterns()
    assert isinstance(patterns1, xr.DataArray)
    assert isinstance(patterns2, xr.DataArray)
    assert isinstance(pvals1, xr.DataArray)
    assert isinstance(pvals2, xr.DataArray)

def test_heterogeneous_patterns(mca_model, test_DataArray):
    mca_model.fit(test_DataArray, test_DataArray, dims='time')
    patterns1, patterns2, pvals1, pvals2 = mca_model.heterogeneous_patterns()
    assert isinstance(patterns1, xr.DataArray)
    assert isinstance(patterns2, xr.DataArray)
    assert isinstance(pvals1, xr.DataArray)
    assert isinstance(pvals2, xr.DataArray)

def test_compute(mca_model, test_DaskDataArray):
    mca_model.fit(test_DaskDataArray, test_DaskDataArray, ('time'))
    assert isinstance(mca_model._singular_values.data, da.Array)
    assert isinstance(mca_model._explained_variance.data, da.Array)
    mca_model.compute()
    assert isinstance(mca_model._singular_values.data, np.ndarray)
    assert isinstance(mca_model._explained_variance.data, np.ndarray)





