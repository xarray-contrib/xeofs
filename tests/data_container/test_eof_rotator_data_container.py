import pytest
import numpy as np
import xarray as xr
from dask.array import Array as DaskArray  # type: ignore

from xeofs.data_container.eof_rotator_data_container import EOFRotatorDataContainer


def test_init():
    """Test the initialization of the EOFRotatorDataContainer."""
    container = EOFRotatorDataContainer()
    assert container._rotation_matrix is None
    assert container._phi_matrix is None
    assert container._modes_sign is None


def test_set_data(
    sample_input_data,
    sample_components,
    sample_scores,
    sample_exp_var,
    sample_rotation_matrix,
    sample_phi_matrix,
    sample_modes_sign,
):
    """Test the set_data() method of EOFRotatorDataContainer."""
    total_variance = sample_exp_var.sum()
    idx_modes_sorted = sample_exp_var.argsort()[::-1]
    container = EOFRotatorDataContainer()
    container.set_data(
        sample_input_data,
        sample_components,
        sample_scores,
        sample_exp_var,
        total_variance,
        idx_modes_sorted,
        sample_modes_sign,
        sample_rotation_matrix,
        sample_phi_matrix,
    )
    assert container._input_data is sample_input_data
    assert container._components is sample_components
    assert container._scores is sample_scores
    assert container._explained_variance is sample_exp_var
    assert container._total_variance is total_variance
    assert container._idx_modes_sorted is idx_modes_sorted
    assert container._modes_sign is sample_modes_sign
    assert container._rotation_matrix is sample_rotation_matrix
    assert container._phi_matrix is sample_phi_matrix


def test_no_data():
    """Test the data accessors without data for EOFRotatorDataContainer."""
    container = EOFRotatorDataContainer()
    with pytest.raises(ValueError):
        container.input_data
    with pytest.raises(ValueError):
        container.components
    with pytest.raises(ValueError):
        container.scores
    with pytest.raises(ValueError):
        container.explained_variance
    with pytest.raises(ValueError):
        container.total_variance
    with pytest.raises(ValueError):
        container.idx_modes_sorted
    with pytest.raises(ValueError):
        container.modes_sign
    with pytest.raises(ValueError):
        container.rotation_matrix
    with pytest.raises(ValueError):
        container.phi_matrix
    with pytest.raises(ValueError):
        container.set_attrs({"test": 1})
    with pytest.raises(ValueError):
        container.compute()
