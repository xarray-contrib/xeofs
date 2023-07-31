import pytest
import numpy as np
import xarray as xr
from dask.array import Array as DaskArray  # type: ignore

from xeofs.data_container.eof_rotator_data_container import ComplexEOFRotatorDataContainer


def test_complex_rotator_init():
    '''Test the initialization of the ComplexEOFRotatorDataContainer.'''
    container = ComplexEOFRotatorDataContainer()
    assert container._rotation_matrix is None
    assert container._phi_matrix is None
    assert container._modes_sign is None

def test_complex_rotator_set_data(sample_input_data, sample_components, sample_scores, sample_exp_var, sample_rotation_matrix, sample_phi_matrix, sample_modes_sign):
    '''Test the set_data() method of ComplexEOFRotatorDataContainer.'''
    total_variance = sample_exp_var.sum()
    idx_modes_sorted = sample_exp_var.argsort()[::-1]
    container = ComplexEOFRotatorDataContainer()
    container.set_data(
        sample_input_data,
        sample_components,
        sample_scores,
        sample_exp_var,
        total_variance,
        idx_modes_sorted,
        sample_rotation_matrix,
        sample_phi_matrix,
        sample_modes_sign,
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
