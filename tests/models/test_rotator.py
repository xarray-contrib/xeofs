import pytest

from xeofs.models import EOF, ComplexEOF, MCA, ComplexMCA
from xeofs.models import EOFRotator, ComplexEOFRotator, MCARotator, ComplexMCARotator
from xeofs.models.rotator_factory import RotatorFactory

# RotatorFactory should be imported from its module
# from module import RotatorFactory


def test_rotator_factory_init():
    factory = RotatorFactory(n_modes=3, power=2, max_iter=1000, rtol=1e-8)
    assert factory.params == {"n_modes": 3, "power": 2, "max_iter": 1000, "rtol": 1e-8}
    assert factory._valid_types == (EOF, ComplexEOF, MCA, ComplexMCA)


def test_create_rotator_EOF():
    factory = RotatorFactory(n_modes=3, power=2, max_iter=1000, rtol=1e-8)
    EOF_instance = EOF()
    rotator = factory.create_rotator(EOF_instance)
    assert isinstance(rotator, EOFRotator)


def test_create_rotator_ComplexEOF():
    factory = RotatorFactory(n_modes=3, power=2, max_iter=1000, rtol=1e-8)
    ComplexEOF_instance = ComplexEOF()  # creating instance of the mock class
    rotator = factory.create_rotator(ComplexEOF_instance)
    assert isinstance(rotator, ComplexEOFRotator)


def test_create_rotator_MCA():
    factory = RotatorFactory(n_modes=3, power=2, max_iter=1000, rtol=1e-8)
    MCA_instance = MCA()
    rotator = factory.create_rotator(MCA_instance)
    assert isinstance(rotator, MCARotator)


def test_create_rotator_ComplexMCA():
    factory = RotatorFactory(n_modes=3, power=2, max_iter=1000, rtol=1e-8)
    ComplexMCA_instance = ComplexMCA()
    rotator = factory.create_rotator(ComplexMCA_instance)
    assert isinstance(rotator, ComplexMCARotator)


def test_create_rotator_invalid_model():
    factory = RotatorFactory(n_modes=3, power=2, max_iter=1000, rtol=1e-8)
    with pytest.raises(TypeError):
        factory.create_rotator("InvalidType")  # type: ignore
