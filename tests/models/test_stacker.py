import pytest
import xarray as xr
import numpy as np

from xeofs.models.stacker import DataArrayStacker, DataArrayListStacker, DatasetStacker

@pytest.fixture
def simple_da():
    return xr.DataArray(np.random.rand(3, 4, 5), dims=("x", "y", "z"))

@pytest.fixture
def simple_ds():
    return xr.Dataset({'var': (("x", "y", "z"), np.random.rand(3, 4, 5))})

@pytest.fixture
def simple_dal():
    return [xr.DataArray(np.random.rand(3, 4, 5), dims=("x", "y", "z")) for _ in range(3)]


def test_DataArrayStacker(test_DataArray):
    dim_sample = 'time'
    dim_feature = ('y', 'x')
    stacker = DataArrayStacker()
    stacker.fit(test_DataArray, dim_sample, dim_feature)
    stacked = stacker.transform(test_DataArray)
    assert stacked.dims == ('sample', 'feature')
    with pytest.raises(ValueError):
        stacker.transform(xr.DataArray(np.random.rand(2, 4, 5), dims=("a", "y", "x")))


def test_DataArrayListStacker(test_DataArrayList):
    dim_sample = 'time'
    dim_feature = [('y', 'x')] * 3
    stacker_list = DataArrayListStacker()
    stacker_list.fit(test_DataArrayList, dim_sample, dim_feature)  #type: ignore
    stacked = stacker_list.transform(test_DataArrayList)
    assert stacked.dims == ('sample', 'feature')
    with pytest.raises(ValueError):
        stacker_list.transform([xr.DataArray(np.random.rand(2, 4, 5), dims=("a", "y", "x")) for _ in range(3)])


def test_DatasetStacker(test_Dataset):
    dim_sample = 'time'
    dim_feature = ('y', 'x')
    stacker = DatasetStacker()
    stacker.fit(test_Dataset, dim_sample, dim_feature)
    stacked = stacker.transform(test_Dataset)
    assert stacked.dims == ('feature', 'sample')
    with pytest.raises(ValueError):
        stacker.transform(xr.Dataset({'var': (("a", "y", "x"), np.random.rand(3, 4, 5))}))


def test_DataArrayStacker_unstack(test_DataArray):
    '''Test if the unstacked DataArray has the same coordinates as the original DataArray.'''
    dim_sample = 'time'
    dim_feature = ('y', 'x')
    stacker = DataArrayStacker()
    stacker.fit(test_DataArray, dim_sample, dim_feature)
    stacked = stacker.transform(test_DataArray)

    unstacked = stacker.inverse_transform_data(stacked)
    for dim, coords in test_DataArray.coords.items():
        assert unstacked.coords[dim].size == coords.size, 'Dimension {} has different size.'.format(dim)


def test_DataArrayListStacker_unstack(test_DataArrayList):
    '''Test if the unstacked DataArrays has the same coordinates as the original DataArrays.'''
    dim_sample = 'time'
    dim_feature = [('y', 'x')] * 3
    stacker_list = DataArrayListStacker()
    stacker_list.fit(test_DataArrayList, dim_sample, dim_feature)  #type: ignore
    stacked = stacker_list.transform(test_DataArrayList)

    unstacked = stacker_list.inverse_transform_data(stacked)
    for da_test, da_ref in zip(unstacked, test_DataArrayList):
        for dim, coords in da_ref.coords.items():
            assert da_test.coords[dim].size == coords.size, 'Dimension {} has different size.'.format(dim)


def test_DatasetStacker_unstack(test_Dataset):
    '''Test if the unstacked Dataset has the same coordinates as the original Dataset.'''
    dim_sample = 'time'
    dim_feature = ('y', 'x')
    stacker = DatasetStacker()
    stacker.fit(test_Dataset, dim_sample, dim_feature)
    stacked = stacker.transform(test_Dataset)

    unstacked = stacker.inverse_transform_data(stacked)
    for dim, coords in test_Dataset.coords.items():
        assert unstacked.coords[dim].size == coords.size, 'Dimension {} has different size.'.format(dim)