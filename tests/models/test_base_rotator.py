# content of test_base_rotator.py
import pytest
import numpy as np
import xarray as xr

# Import the classes from your modules
from xeofs.models._base_rotator import _BaseRotator

def test_BaseRotator_init(test_DataArray):
    # Instantiate the BaseRotator class
    base_rotator = _BaseRotator(10, power=2, max_iter=100, rtol=1e-6)

    assert base_rotator._params['n_rot'] == 10
    assert base_rotator._params['power'] == 2
    assert base_rotator._params['max_iter'] == 100
    assert base_rotator._params['rtol'] == 1e-6

def test_BaseRotator_fit_raises_error(test_DataArray):
    base_rotator = _BaseRotator(10)

    with pytest.raises(NotImplementedError):
        base_rotator.fit(test_DataArray)

def test_BaseRotator_transform_raises_error(test_DataArray):
    base_rotator = _BaseRotator(10)

    with pytest.raises(NotImplementedError):
        base_rotator.transform(test_DataArray)

def test_BaseRotator_inverse_transform_raises_error(test_DataArray):
    base_rotator = _BaseRotator(10)

    with pytest.raises(NotImplementedError):
        base_rotator.inverse_transform(test_DataArray)
