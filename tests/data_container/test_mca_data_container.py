import pytest
import xarray as xr
import numpy as np

from xeofs.data_container.mca_data_container import MCADataContainer


def test_mca_data_container_init():
    '''Test the initialization of the MCADataContainer.'''
    data_container = MCADataContainer()
    assert data_container._input_data1 is None
    assert data_container._input_data2 is None
    assert data_container._components1 is None
    assert data_container._components2 is None
    assert data_container._scores1 is None
    assert data_container._scores2 is None
    assert data_container._squared_covariance is None
    assert data_container._total_squared_covariance is None
    assert data_container._idx_modes_sorted is None
    assert data_container._norm1 is None
    assert data_container._norm2 is None


def test_mca_data_container_set_data(sample_input_data, sample_components, sample_scores, sample_squared_covariance, sample_total_squared_covariance, sample_idx_modes_sorted, sample_norm):
    '''Test the set_data() method of MCADataContainer.'''
    data_container = MCADataContainer()
    data_container.set_data(sample_input_data, sample_input_data, sample_components, sample_components, sample_scores, sample_scores, sample_squared_covariance, sample_total_squared_covariance, sample_idx_modes_sorted, sample_norm, sample_norm)
    
    assert data_container._input_data1 is sample_input_data
    assert data_container._input_data2 is sample_input_data
    assert data_container._components1 is sample_components
    assert data_container._components2 is sample_components
    assert data_container._scores1 is sample_scores
    assert data_container._scores2 is sample_scores
    assert data_container._squared_covariance is sample_squared_covariance
    assert data_container._total_squared_covariance is sample_total_squared_covariance
    assert data_container._idx_modes_sorted is sample_idx_modes_sorted
    assert data_container._norm1 is sample_norm
    assert data_container._norm2 is sample_norm


def test_mca_data_container_no_data():
    '''Test the data accessors without data in MCADataContainer.'''
    data_container = MCADataContainer()
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
        data_container.squared_covariance
    with pytest.raises(ValueError):
        data_container.total_squared_covariance
    with pytest.raises(ValueError):
        data_container.squared_covariance_fraction
    with pytest.raises(ValueError):
        data_container.singular_values
    with pytest.raises(ValueError):
        data_container.covariance_fraction
    with pytest.raises(ValueError):
        data_container.idx_modes_sorted
    with pytest.raises(ValueError):
        data_container.norm1
    with pytest.raises(ValueError):
        data_container.norm2
    with pytest.raises(ValueError):
        data_container.set_attrs({'test': 1})
    with pytest.raises(ValueError):
        data_container.compute()


def test_mca_data_container_set_attrs(sample_input_data, sample_components, sample_scores, sample_squared_covariance, sample_total_squared_covariance, sample_idx_modes_sorted, sample_norm):
    '''Test the set_attrs() method of MCADataContainer.'''
    data_container = MCADataContainer()
    data_container.set_data(sample_input_data, sample_input_data, sample_components, sample_components, sample_scores, sample_scores, sample_squared_covariance, sample_total_squared_covariance, sample_idx_modes_sorted, sample_norm, sample_norm)
    data_container.set_attrs({'test': 1})
    
    assert data_container.components1.attrs['test'] == 1
    assert data_container.components2.attrs['test'] == 1
    assert data_container.scores1.attrs['test'] == 1
    assert data_container.scores2.attrs['test'] == 1
    assert data_container.squared_covariance.attrs['test'] == 1
    assert data_container.total_squared_covariance.attrs['test'] == 1
    assert data_container.squared_covariance_fraction.attrs['test'] == 1
    assert data_container.singular_values.attrs['test'] == 1
    assert data_container.total_covariance.attrs['test'] == 1
    assert data_container.covariance_fraction.attrs['test'] == 1
    assert data_container.norm1.attrs['test'] == 1
    assert data_container.norm2.attrs['test'] == 1
