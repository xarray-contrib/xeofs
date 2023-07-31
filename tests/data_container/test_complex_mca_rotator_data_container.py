import pytest
import xarray as xr
import numpy as np

from xeofs.data_container.mca_rotator_data_container import (
    ComplexMCARotatorDataContainer,
)
from .test_mca_rotator_data_container import (
    test_mca_rotator_data_container_init,
    test_mca_rotator_data_container_set_data,
    test_mca_rotator_data_container_no_data,
    test_mca_rotator_data_container_set_attrs,
)


def test_complex_mca_rotator_data_container_init():
    """Test the initialization of the ComplexMCARotatorDataContainer."""
    data_container = ComplexMCARotatorDataContainer()
    test_mca_rotator_data_container_init()  # Re-use the test from MCARotatorDataContainer.


def test_complex_mca_rotator_data_container_set_data(
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
    """Test the set_data() method of ComplexMCARotatorDataContainer."""
    data_container = ComplexMCARotatorDataContainer()
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

    test_mca_rotator_data_container_set_data(
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
    )  # Re-use the test from MCARotatorDataContainer.


def test_complex_mca_rotator_data_container_no_data():
    """Test the data accessors without data in ComplexMCARotatorDataContainer."""
    data_container = ComplexMCARotatorDataContainer()
    test_mca_rotator_data_container_no_data()  # Re-use the test from MCARotatorDataContainer.


def test_complex_mca_rotator_data_container_set_attrs(
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
    """Test the set_attrs() method of ComplexMCARotatorDataContainer."""
    data_container = ComplexMCARotatorDataContainer()
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

    test_mca_rotator_data_container_set_attrs(
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
    )  # Re-use the test from MCARotatorDataContainer.
