import warnings

from .cca import CCA

__all__ = ["CCA"]


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
