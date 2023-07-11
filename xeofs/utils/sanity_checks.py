from typing import Sequence, Hashable, Tuple, Any

import xarray as xr

def assert_single_dataarray(da, name):
    """Check if the given object is a DataArray.

    Args:
        da (DataArray): The object to check.
        name (str): The name of the object.

    Raises:
        TypeError: If the object is not a DataArray.
    """
    if not isinstance(da, xr.DataArray):
        raise TypeError(f"{name} must be a DataArray")
    
def assert_list_dataarrays(da_list, name):
    """Check if the given object is a list of DataArrays.

    Args:
        da_list (list): The object to check.
        name (str): The name of the object.

    Raises:
        TypeError: If the object is not a list of DataArrays.
    """
    if not isinstance(da_list, list):
        raise TypeError(f"{name} must be a list of DataArrays")
    for da in da_list:
        assert_single_dataarray(da, name)

def assert_single_dataset(ds, name):
    """Check if the given object is a Dataset.

    Args:
        ds (Dataset): The object to check.
        name (str): The name of the object.

    Raises:
        TypeError: If the object is not a Dataset.
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"{name} must be a Dataset")

def assert_dataarray_or_dataset(da, name):
    """Check if the given object is a DataArray or Dataset.
    
    Args:
        da (DataArray|Dataset): The object to check.
        name (str): The name of the object.
    
    Raises:
        TypeError: If the object is not a DataArray or Dataset.
    """
    if not isinstance(da, (xr.DataArray, xr.Dataset)):
        raise TypeError(f"{name} must be either a DataArray or Dataset")


def ensure_tuple(arg: Any) -> Tuple[str]:
    # Check for invalid types
    if not isinstance(arg, (str, tuple, list)):
        raise TypeError(f"Invalid input type: {type(arg).__name__}")

    # Check for invalid sequence elements
    if isinstance(arg, (tuple, list)) and not all(isinstance(item, str) for item in arg):
        raise TypeError("Invalid sequence element type. All elements should be strings")

    if isinstance(arg, tuple):
        return arg
    elif isinstance(arg, list):
        return tuple(arg)
    else:
        return (arg,)