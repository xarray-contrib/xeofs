import pytest
from unittest.mock import Mock

from xeofs.models import EOF, ComplexEOF
from xeofs.models.rotator import RotatorFactory, EOFRotator, ComplexEOFRotator

# RotatorFactory should be imported from its module
# from module import RotatorFactory

def test_rotator_factory_init():
    factory = RotatorFactory(n_rot=3, power=2, max_iter=1000, rtol=1e-8)
    assert factory.params == {'n_rot': 3, 'power': 2, 'max_iter': 1000, 'rtol': 1e-8}
    assert factory._valid_types == (EOF, ComplexEOF)

def test_create_rotator_EOF():
    factory = RotatorFactory(n_rot=3, power=2, max_iter=1000, rtol=1e-8)
    EOF_instance = EOF()
    rotator = factory.create_rotator(EOF_instance)
    assert isinstance(rotator, EOFRotator)

def test_create_rotator_ComplexEOF():
    factory = RotatorFactory(n_rot=3, power=2, max_iter=1000, rtol=1e-8)
    ComplexEOF_instance = ComplexEOF()  # creating instance of the mock class
    rotator = factory.create_rotator(ComplexEOF_instance)
    assert isinstance(rotator, ComplexEOFRotator)

def test_create_rotator_invalid_model():
    factory = RotatorFactory(n_rot=3, power=2, max_iter=1000, rtol=1e-8)
    with pytest.raises(TypeError):
        factory.create_rotator("InvalidType")  # type: ignore
