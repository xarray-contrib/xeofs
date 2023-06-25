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


def test_DataArrayStacker(simple_da):
    stacker = DataArrayStacker()
    stacker.fit(simple_da, 'x', ['y', 'z'])
    stacked = stacker.stack_data(simple_da)
    assert stacked.dims == ('sample', 'feature')
    with pytest.raises(ValueError):
        stacker.stack_data(xr.DataArray(np.random.rand(2, 4, 5), dims=("a", "y", "z")))


def test_DataArrayListStacker(simple_dal):
    stacker_list = DataArrayListStacker()
    stacker_list.fit(simple_dal, 'x', [['y', 'z'], ['y', 'z'], ['y', 'z']])
    stacked = stacker_list.stack_data(simple_dal)
    assert all(stacked_da.dims == ('sample', 'feature') for stacked_da in stacked)
    with pytest.raises(ValueError):
        stacker_list.stack_data([xr.DataArray(np.random.rand(2, 4, 5), dims=("a", "y", "z")) for _ in range(3)])


def test_DatasetStacker(simple_ds):
    stacker = DatasetStacker()
    stacker.fit(simple_ds, 'x', ['y', 'z'])
    stacked = stacker.stack_data(simple_ds)
    assert stacked.dims == ('feature', 'sample')
    with pytest.raises(ValueError):
        stacker.stack_data(xr.Dataset({'var': (("a", "y", "z"), np.random.rand(3, 4, 5))}))


def test_DataArrayStacker_unstack(simple_da):
    stacker = DataArrayStacker()
    stacker.fit(simple_da, 'x', ['y', 'z'])
    stacked = stacker.stack_data(simple_da)

    unstacked = stacker.unstack_data(stacked)
    assert unstacked.dims == simple_da.dims


def test_DataArrayListStacker_unstack(simple_dal):
    stacker_list = DataArrayListStacker()
    stacker_list.fit(simple_dal, 'x', [['y', 'z'], ['y', 'z'], ['y', 'z']])
    stacked = stacker_list.stack_data(simple_dal)

    unstacked = stacker_list.unstack_data(stacked)
    assert all(unstacked_da.dims == da.dims for unstacked_da, da in zip(unstacked, simple_dal))


def test_DatasetStacker_unstack(simple_ds):
    stacker = DatasetStacker()
    stacker.fit(simple_ds, 'x', ['y', 'z'])
    stacked = stacker.stack_data(simple_ds)

    unstacked = stacker.unstack_data(stacked)
    assert unstacked.dims == simple_ds.dims