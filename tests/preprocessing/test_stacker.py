import pytest
import xarray as xr
import numpy as np

from xeofs.preprocessing.stacker import DataArrayStacker, DataArrayListStacker, DatasetStacker

@pytest.fixture
def simple_da():
    return xr.DataArray(np.random.rand(3, 4, 5), dims=("x", "y", "z"))

@pytest.fixture
def simple_ds():
    return xr.Dataset({'var': (("x", "y", "z"), np.random.rand(3, 4, 5))})

@pytest.fixture
def simple_dal():
    return [xr.DataArray(np.random.rand(3, 4, 5), dims=("x", "y", "z")) for _ in range(3)]


@pytest.mark.parametrize('dim_sample, dim_feature', [
    ('time', ('y', 'x')),
    (('x', 'y'), 'time'),
    ])
def test_DataArrayStacker(dim_sample, dim_feature, test_DataArray):
    stacker = DataArrayStacker()
    stacker.fit(test_DataArray, dim_sample, dim_feature)
    stacked = stacker.transform(test_DataArray)

    # Check if the dimensions are correct
    assert stacked.dims == ('sample', 'feature'), 'Stacked dimensions are not correct.'

    # Check if the data is preserved
    assert stacked.notnull().sum() == test_DataArray.notnull().sum(), 'Stacked data is not correct.'

    with pytest.raises(ValueError):
        stacker.transform(xr.DataArray(np.random.rand(2, 4, 5), dims=("a", "y", "x")))


@pytest.mark.parametrize('dim_sample, dim_feature', [
    ('time', ('y', 'x')),
    (('x', 'y'), 'time'),
    ])
def test_DatasetStacker(dim_sample, dim_feature, test_Dataset):
    stacker = DatasetStacker()
    stacker.fit(test_Dataset, dim_sample, dim_feature)
    stacked = stacker.transform(test_Dataset)

    # Check if the dimensions are correct
    assert stacked.dims == ('feature', 'sample')

    # Check if the data is preserved
    assert stacked.notnull().sum() == test_Dataset.notnull().sum(), 'Stacked data is not correct.'

    with pytest.raises(ValueError):
        stacker.transform(xr.Dataset({'var': (("a", "y", "x"), np.random.rand(3, 4, 5))}))  # type: ignore


@pytest.mark.parametrize('dim_sample, dim_feature', [
    ('time', 3*[('y', 'x')]),
    (('x', 'y'), 3*['time']),
    ])
def test_DataArrayListStacker(dim_sample, dim_feature, test_DataArrayList):
    stacker_list = DataArrayListStacker()
    stacker_list.fit(test_DataArrayList, dim_sample, dim_feature)  #type: ignore
    stacked = stacker_list.transform(test_DataArrayList)

    # Check if the dimensions are correct
    assert stacked.dims == ('sample', 'feature')

    # Check if the data is preserved
    assert stacked.notnull().sum() == sum([da.notnull().sum() for da in test_DataArrayList]), 'Stacked data is not correct.'

    with pytest.raises(ValueError):
        stacker_list.transform([xr.DataArray(np.random.rand(2, 4, 5), dims=("a", "y", "x")) for _ in range(3)])



@pytest.mark.parametrize('dim_sample, dim_feature', [
    ('time', ('y', 'x')),
    (('x', 'y'), 'time'),
    ])
def test_DataArrayStacker_unstack(dim_sample, dim_feature, test_DataArray):
    '''Test if the unstacked DataArray has the same coordinates as the original DataArray.'''
    stacker = DataArrayStacker()
    stacker.fit(test_DataArray, dim_sample, dim_feature)
    stacked = stacker.transform(test_DataArray)

    unstacked = stacker.inverse_transform_data(stacked)
    for dim, coords in test_DataArray.coords.items():
        assert unstacked.coords[dim].size == coords.size, 'Dimension {} has different size.'.format(dim)

@pytest.mark.parametrize('dim_sample, dim_feature', [
    ('time', ('y', 'x')),
    (('x', 'y'), 'time'),
    ])
def test_DatasetStacker_unstack(dim_sample, dim_feature, test_Dataset):
    '''Test if the unstacked Dataset has the same coordinates as the original Dataset.'''
    stacker = DatasetStacker()
    stacker.fit(test_Dataset, dim_sample, dim_feature)
    stacked = stacker.transform(test_Dataset)

    unstacked = stacker.inverse_transform_data(stacked)
    for dim, coords in test_Dataset.coords.items():
        assert unstacked.coords[dim].size == coords.size, 'Dimension {} has different size.'.format(dim)

@pytest.mark.parametrize('dim_sample, dim_feature', [
    ('time', 3*[('y', 'x')]),
    (('x', 'y'), 3*[('time',)]),
    ])
def test_DataArrayListStacker_unstack(dim_sample, dim_feature, test_DataArrayList):
    '''Test if the unstacked DataArrays has the same coordinates as the original DataArrays.'''
    stacker_list = DataArrayListStacker()
    stacker_list.fit(test_DataArrayList, dim_sample, dim_feature)  #type: ignore
    stacked = stacker_list.transform(test_DataArrayList)

    unstacked = stacker_list.inverse_transform_data(stacked)
    for da_test, da_ref in zip(unstacked, test_DataArrayList):
        for dim, coords in da_ref.coords.items():
            assert da_test.coords[dim].size == coords.size, 'Dimension {} has different size.'.format(dim)



@pytest.mark.parametrize('dim_sample, dim_feature', [
    ('time', ('y', 'x')),
    (('x', 'y'), ('time',)),
    ])
def test_DataArrayStacker_transform_error(dim_sample, dim_feature, test_DataArray):
    '''Transform method raises an error if the coordinates of the input data are different from the coordinates of the fitted data.'''
    ncoords = test_DataArray.coords[dim_feature[0]].size
    other = test_DataArray.assign_coords({dim_feature[0]: np.arange(ncoords)})

    stacker = DataArrayStacker()
    stacker.fit(test_DataArray, dim_sample, dim_feature)

    with pytest.raises(ValueError):
        stacker.transform(other)
 