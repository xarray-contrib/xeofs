import warnings

from .cca import CCA
from .eeof import ExtendedEOF
from .eof import EOF, HilbertEOF
from .eof_rotator import EOFRotator, HilbertEOFRotator
from .gwpca import GWPCA
from .mca import MCA, HilbertMCA
from .mca_rotator import HilbertMCARotator, MCARotator
from .opa import OPA
from .rotator_factory import RotatorFactory
from .sparse_pca import SparsePCA

__all__ = [
    "EOF",
    "HilbertEOF",
    "ExtendedEOF",
    "EOFRotator",
    "HilbertEOFRotator",
    "OPA",
    "GWPCA",
    "MCA",
    "HilbertMCA",
    "MCARotator",
    "HilbertMCARotator",
    "CCA",
    "RotatorFactory",
    "SparsePCA",
]


DEPRECATED_NAMES = [
    ("ComplexEOF", "HilbertEOF"),
    ("ComplexMCA", "HilbertMCA"),
    ("ComplexEOFRotator", "HilbertEOFRotator"),
    ("ComplexMCARotator", "HilbertMCARotator"),
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
