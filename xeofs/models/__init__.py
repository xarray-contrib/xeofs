import warnings

from .cca import CCA
from .eeof import ExtendedEOF
from .eof import EOF, ComplexEOF, HilbertEOF
from .eof_rotator import ComplexEOFRotator, EOFRotator, HilbertEOFRotator
from .gwpca import GWPCA
from .mca import MCA, ComplexMCA, HilbertMCA
from .mca_rotator import ComplexMCARotator, HilbertMCARotator, MCARotator
from .opa import OPA
from .rotator_factory import RotatorFactory
from .sparse_pca import SparsePCA

__all__ = [
    "EOF",
    "ComplexEOF",
    "HilbertEOF",
    "ExtendedEOF",
    "EOFRotator",
    "ComplexEOFRotator",
    "HilbertEOFRotator",
    "OPA",
    "GWPCA",
    "MCA",
    "ComplexMCA",
    "HilbertMCA",
    "MCARotator",
    "ComplexMCARotator",
    "HilbertMCARotator",
    "CCA",
    "RotatorFactory",
    "SparsePCA",
]


DEPRECATED_NAMES = [
    # ("OldClass", "NewClass"),
]


def __dir__():
    return sorted(__all__ + [names[0] for names in DEPRECATED_NAMES])


def __getattr__(name):
    for old_name, new_name in DEPRECATED_NAMES:
        if name == old_name:
            msg = (
                f"Class '{old_name}' is deprecated and will be renamed to '{new_name}' in the next major release. "
                f"In that release, '{old_name}' will refer to a different class. "
                f"Please switch to '{new_name}' to maintain compatibility."
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return globals()[new_name]
    raise AttributeError(f"module {__name__} has no attribute {name}")
