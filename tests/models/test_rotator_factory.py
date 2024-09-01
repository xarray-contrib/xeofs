import pytest

from xeofs.cross import MCA, HilbertMCA, HilbertMCARotator, MCARotator
from xeofs.rotator_factory import RotatorFactory
from xeofs.single import EOF, EOFRotator, HilbertEOF, HilbertEOFRotator

# RotatorFactory should be imported from its module
# from module import RotatorFactory


def test_rotator_factory_init():
    factory = RotatorFactory(n_modes=3, power=2, max_iter=1000, rtol=1e-8)
    assert factory.params == {"n_modes": 3, "power": 2, "max_iter": 1000, "rtol": 1e-8}
    assert factory._valid_types == (EOF, HilbertEOF, MCA, HilbertMCA)


def test_create_rotator_EOF():
    factory = RotatorFactory(n_modes=3, power=2, max_iter=1000, rtol=1e-8)
    EOF_instance = EOF()
    rotator = factory.create_rotator(EOF_instance)
    assert isinstance(rotator, EOFRotator)


def test_create_rotator_HilbertEOF():
    factory = RotatorFactory(n_modes=3, power=2, max_iter=1000, rtol=1e-8)
    HilbertEOF_instance = HilbertEOF()  # creating instance of the mock class
    rotator = factory.create_rotator(HilbertEOF_instance)
    assert isinstance(rotator, HilbertEOFRotator)


def test_create_rotator_MCA():
    factory = RotatorFactory(n_modes=3, power=2, max_iter=1000, rtol=1e-8)
    MCA_instance = MCA()
    rotator = factory.create_rotator(MCA_instance)
    assert isinstance(rotator, MCARotator)


def test_create_rotator_HilbertMCA():
    factory = RotatorFactory(n_modes=3, power=2, max_iter=1000, rtol=1e-8)
    HilbertMCA_instance = HilbertMCA()
    rotator = factory.create_rotator(HilbertMCA_instance)
    assert isinstance(rotator, HilbertMCARotator)


def test_create_rotator_invalid_model():
    factory = RotatorFactory(n_modes=3, power=2, max_iter=1000, rtol=1e-8)
    with pytest.raises(TypeError):
        factory.create_rotator("InvalidType")  # type: ignore
