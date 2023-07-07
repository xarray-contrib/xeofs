# content of test_base_rotator.py
import pytest
import numpy as np
import xarray as xr

# Import the classes from your modules
from xeofs.models._base_rotator import _BaseRotator

def test_BaseRotator_init(mock_data_array):
    # Instantiate the BaseRotator class
    base_rotator = _BaseRotator(10, power=2, max_iter=100, rtol=1e-6)

    assert base_rotator._params['n_modes'] == 10
    assert base_rotator._params['power'] == 2
    assert base_rotator._params['max_iter'] == 100
    assert base_rotator._params['rtol'] == 1e-6

def test_BaseRotator_fit_raises_error(mock_data_array):
    base_rotator = _BaseRotator(10)

    with pytest.raises(NotImplementedError):
        base_rotator.fit(mock_data_array)

def test_BaseRotator_transform_raises_error(mock_data_array):
    base_rotator = _BaseRotator(10)

    with pytest.raises(NotImplementedError):
        base_rotator.transform(mock_data_array)

def test_BaseRotator_inverse_transform_raises_error(mock_data_array):
    base_rotator = _BaseRotator(10)

    with pytest.raises(NotImplementedError):
        base_rotator.inverse_transform(mock_data_array)
