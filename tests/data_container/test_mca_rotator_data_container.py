import pytest
import xarray as xr
import numpy as np

from xeofs.data_container.mca_rotator_data_container import MCARotatorDataContainer
from .test_mca_data_container import (
    test_mca_data_container_init,
    test_mca_data_container_set_data,
    test_mca_data_container_no_data,
    test_mca_data_container_set_attrs,
)

"""
The idea here is to reuse tests from MCADataContainer in MCARotatorDataContainer 
and then tests from MCARotatorDataContainer in ComplexMCARotatorDataContainer, 
while also testing the new functionality of each class. This way, we ensure that 
inherited behavior still works as expected in subclasses. If some new tests fail, 
we'll know it's due to the new functionality and not something inherited.
"""


def test_mca_rotator_data_container_init():
    """Test the initialization of the MCARotatorDataContainer."""
    data_container = MCARotatorDataContainer()
    test_mca_data_container_init()  # Re-use the test from MCADataContainer.
    assert data_container._rotation_matrix is None
    assert data_container._phi_matrix is None
    assert data_container._modes_sign is None


def test_mca_rotator_data_container_set_data(
    sample_input_data,
    sample_components,
    sample_scores,
    sample_squared_covariance,
    sample_total_squared_covariance,
    sample_idx_modes_sorted,
    sample_norm,
    sample_rotation_matrix,
    sample_phi_matrix,
    sample_modes_sign,
):
    """Test the set_data() method of MCARotatorDataContainer."""
    data_container = MCARotatorDataContainer()
    data_container.set_data(
        sample_input_data,
        sample_input_data,
        sample_components,
        sample_components,
        sample_scores,
        sample_scores,
        sample_squared_covariance,
        sample_total_squared_covariance,
        sample_idx_modes_sorted,
        sample_modes_sign,
        sample_norm,
        sample_norm,
        sample_rotation_matrix,
        sample_phi_matrix,
    )

    test_mca_data_container_set_data(
        sample_input_data,
        sample_components,
        sample_scores,
        sample_squared_covariance,
        sample_total_squared_covariance,
        sample_idx_modes_sorted,
        sample_norm,
    )  # Re-use the test from MCADataContainer.
    assert data_container._rotation_matrix is sample_rotation_matrix
    assert data_container._phi_matrix is sample_phi_matrix
    assert data_container._modes_sign is sample_modes_sign


def test_mca_rotator_data_container_no_data():
    """Test the data accessors without data in MCARotatorDataContainer."""
    data_container = MCARotatorDataContainer()
    test_mca_data_container_no_data()  # Re-use the test from MCADataContainer.
    with pytest.raises(ValueError):
        data_container.rotation_matrix
    with pytest.raises(ValueError):
        data_container.phi_matrix
    with pytest.raises(ValueError):
        data_container.modes_sign


def test_mca_rotator_data_container_set_attrs(
    sample_input_data,
    sample_components,
    sample_scores,
    sample_squared_covariance,
    sample_total_squared_covariance,
    sample_idx_modes_sorted,
    sample_norm,
    sample_rotation_matrix,
    sample_phi_matrix,
    sample_modes_sign,
):
    """Test the set_attrs() method of MCARotatorDataContainer."""
    data_container = MCARotatorDataContainer()
    data_container.set_data(
        sample_input_data,
        sample_input_data,
        sample_components,
        sample_components,
        sample_scores,
        sample_scores,
        sample_squared_covariance,
        sample_total_squared_covariance,
        sample_idx_modes_sorted,
        sample_modes_sign,
        sample_norm,
        sample_norm,
        sample_rotation_matrix,
        sample_phi_matrix,
    )
    data_container.set_attrs({"test": 1})

    test_mca_data_container_set_attrs(
        sample_input_data,
        sample_components,
        sample_scores,
        sample_squared_covariance,
        sample_total_squared_covariance,
        sample_idx_modes_sorted,
        sample_norm,
    )  # Re-use the test from MCADataContainer.
    assert data_container.rotation_matrix.attrs["test"] == 1
    assert data_container.phi_matrix.attrs["test"] == 1
    assert data_container.modes_sign.attrs["test"] == 1
