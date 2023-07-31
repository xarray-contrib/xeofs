import pytest
import xarray as xr
import numpy as np

from xeofs.data_container._base_cross_model_data_container import _BaseCrossModelDataContainer


def test_init():
    '''Test the initialization of the BaseCrossModelDataContainer.'''
    data_container = _BaseCrossModelDataContainer()
    assert data_container._input_data1 is None
    assert data_container._input_data2 is None
    assert data_container._components1 is None
    assert data_container._components2 is None
    assert data_container._scores1 is None
    assert data_container._scores2 is None

def test_set_data(sample_input_data, sample_components, sample_scores):
    '''Test the set_data() method.'''
    data_container = _BaseCrossModelDataContainer()
    data_container.set_data(sample_input_data, sample_input_data, sample_components, sample_components, sample_scores, sample_scores)
    assert data_container._input_data1 is sample_input_data
    assert data_container._input_data2 is sample_input_data
    assert data_container._components1 is sample_components
    assert data_container._components2 is sample_components
    assert data_container._scores1 is sample_scores
    assert data_container._scores2 is sample_scores

def test_no_data():
    '''Test the data accessors without data.'''
    data_container = _BaseCrossModelDataContainer()
    with pytest.raises(ValueError):
        data_container.input_data1
    with pytest.raises(ValueError):
        data_container.input_data2
    with pytest.raises(ValueError):
        data_container.components1
    with pytest.raises(ValueError):
        data_container.components2
    with pytest.raises(ValueError):
        data_container.scores1
    with pytest.raises(ValueError):
        data_container.scores2
    with pytest.raises(ValueError):
        data_container.set_attrs({'test': 1})
    with pytest.raises(ValueError):
        data_container.compute()

def test_set_attrs(sample_input_data, sample_components, sample_scores):
    '''Test the set_attrs() method.'''
    data_container = _BaseCrossModelDataContainer()
    data_container.set_data(sample_input_data, sample_input_data, sample_components, sample_components, sample_scores, sample_scores)
    data_container.set_attrs({'test': 1})
    assert data_container.components1.attrs['test'] == 1
    assert data_container.components2.attrs['test'] == 1
    assert data_container.scores1.attrs['test'] == 1
    assert data_container.scores2.attrs['test'] == 1
