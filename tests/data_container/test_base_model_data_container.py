import pytest
import xarray as xr
import numpy as np

from xeofs.data_container._base_model_data_container import _BaseModelDataContainer


def test_init():
    """Test the initialization of the BaseModelDataContainer."""
    data_container = _BaseModelDataContainer()
    assert data_container._input_data is None
    assert data_container._components is None
    assert data_container._scores is None


def test_set_data(sample_input_data, sample_components, sample_scores):
    """Test the set_data() method."""
    data_container = _BaseModelDataContainer()
    data_container.set_data(sample_input_data, sample_components, sample_scores)
    assert data_container._input_data is sample_input_data
    assert data_container._components is sample_components
    assert data_container._scores is sample_scores


def test_no_data():
    """Test the data accessors without data."""
    data_container = _BaseModelDataContainer()
    with pytest.raises(ValueError):
        data_container.input_data
    with pytest.raises(ValueError):
        data_container.components
    with pytest.raises(ValueError):
        data_container.scores
    with pytest.raises(ValueError):
        data_container.set_attrs({"test": 1})
    with pytest.raises(ValueError):
        data_container.compute()


def test_set_attrs(sample_input_data, sample_components, sample_scores):
    """Test the set_attrs() method."""
    data_container = _BaseModelDataContainer()
    data_container.set_data(sample_input_data, sample_components, sample_scores)
    data_container.set_attrs({"test": 1})
    assert data_container.components.attrs["test"] == 1
    assert data_container.scores.attrs["test"] == 1
