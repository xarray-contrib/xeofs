from typing import Sequence, Hashable, Tuple, TypeVar, List, Any

import numpy as np
import xarray as xr

from .sanity_checks import convert_to_dim_type
from .data_types import (
    Dims,
    DimsList,
    DaskArray,
    Data,
    DataVar,
    DataArray,
    DataSet,
    DataList,
)
from .constants import VALID_LATITUDE_NAMES

T = TypeVar("T")


def unwrap_singleton_list(input_list: List[T]) -> T | List[T]:
    if len(input_list) == 1:
        return input_list[0]
    else:
        return input_list


def data_is_dask(data: DataArray | DataSet | DataList) -> bool:
    """Check if the given data is backed by a dask array."""

    # If data is a DataArray, check its underlying data type
    if isinstance(data, DataArray):
        return isinstance(data.data, DaskArray)

    # If data is a DataSet, recursively check all contained DataArrays
    if isinstance(data, DataSet):
        return any(data_is_dask(da) for da in data.data_vars.values())

    # If data is a list, recursively check each element in the list
    if isinstance(data, list):
        return any(data_is_dask(da) for da in data)

    # If none of the above, the data type is unrecognized
    raise ValueError("unrecognized data type.")


def process_parameter(
    parameter_name: str, parameter, default, n_data: int
) -> List[Any]:
    if parameter is None:
        return convert_to_list(default) * n_data
    elif isinstance(parameter, (list, tuple)):
        _check_parameter_number(parameter_name, parameter, n_data)
        return convert_to_list(parameter)
    else:
        return convert_to_list(parameter) * n_data


def convert_to_list(data: T | List[T] | Tuple[T]) -> List[T]:
    if isinstance(data, list):
        return data
    elif isinstance(data, tuple):
        return list(data)
    else:
        return list([data])


def _check_parameter_number(parameter_name: str, parameter, n_data: int):
    if len(parameter) != n_data:
        raise ValueError(
            f"number of data objects passed should match number of parameter {parameter_name}"
            f"len(data objects)={n_data} and "
            f"len({parameter_name})={len(parameter)}"
        )


def feature_ones_like(data: DataVar, feature_dims: Dims) -> DataVar:
    if isinstance(data, xr.DataArray):
        valid_dims = set(data.dims) & set(feature_dims)
        feature_coords = {dim: data[dim] for dim in valid_dims}
        shape = tuple(coords.size for coords in feature_coords.values())
        return xr.DataArray(
            np.ones(shape, dtype=float),
            dims=tuple(valid_dims),
            coords=feature_coords,
        )
    elif isinstance(data, xr.Dataset):
        return xr.Dataset(
            {
                var: feature_ones_like(da, feature_dims)
                for var, da in data.data_vars.items()
            }
        )
    else:
        raise TypeError(
            "Invalid input type: {:}. Expected one of the following: DataArray or Dataset".format(
                type(data).__name__
            )
        )


def compute_sqrt_cos_lat_weights(data: DataVar, feature_dims: Dims) -> DataVar:
    """Compute the square root of cosine of latitude weights.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        Data to be scaled.
    dim : sequence of hashable
        Dimensions along which the data is considered to be a feature.

    Returns
    -------
    xr.DataArray | xr.Dataset
        Square root of cosine of latitude weights.

    """

    if isinstance(data, xr.DataArray):
        lat_dim = extract_latitude_dimension(feature_dims)

        latitudes = data.coords[lat_dim]
        weights = sqrt_cos_lat_weights(latitudes)
        # Features that cannot be associated to a latitude receive a weight of 1
        # weights = weights.where(weights.notnull(), 1)
        weights.name = "coslat_weights"
        return weights
    elif isinstance(data, xr.Dataset):
        return xr.Dataset(
            {
                var: compute_sqrt_cos_lat_weights(da, feature_dims)
                for var, da in data.data_vars.items()
            }
        )

    else:
        raise TypeError(
            "Invalid input type: {:}. Expected one of the following: DataArray".format(
                type(data).__name__
            )
        )


def extract_latitude_dimension(feature_dims: Dims) -> Hashable:
    # Find latitude coordinate
    lat_dim = set(feature_dims) & set(VALID_LATITUDE_NAMES)

    if len(lat_dim) == 0:
        raise ValueError(
            "No latitude coordinate was found to compute coslat weights. Must be one of the following: {:}".format(
                VALID_LATITUDE_NAMES
            )
        )
    elif len(lat_dim) == 1:
        return lat_dim.pop()
    else:
        raise ValueError(
            f"Found ambiguous latitude dimensions: {lat_dim}. Only ONE of the following is allowed for computing coslat weights: {VALID_LATITUDE_NAMES}"
        )


def get_dims(
    data: DataList,
    sample_dims: Hashable | Sequence[Hashable],
) -> Tuple[Dims, DimsList]:
    """Extracts the dimensions of a DataArray or Dataset that are not included in the sample dimensions.

    Parameters:
    ------------
    data: xr.DataArray or xr.Dataset or list of xr.DataArray
        Input data.
    sample_dims: Hashable or Sequence[Hashable] or List[Sequence[Hashable]]
        Sample dimensions.

    Returns:
    ---------
    sample_dims: Tuple[Hashable]
        Sample dimensions.
    feature_dims: Tuple[Hashable]
        Feature dimensions.

    """
    # Check for invalid types
    if isinstance(data, list):
        sample_dims = convert_to_dim_type(sample_dims)
        feature_dims: DimsList = [_get_feature_dims(da, sample_dims) for da in data]
        return sample_dims, feature_dims
    else:
        err_message = f"Invalid input type: {type(data).__name__}. Expected one of "
        err_message += f"of the following: list of DataArrays or Datasets."
        raise TypeError(err_message)


def _get_feature_dims(data: DataArray | DataSet, sample_dims: Dims) -> Dims:
    """Extracts the dimensions of a DataArray that are not included in the sample dimensions.


    Parameters:
    ------------
    data: xr.DataArray or xr.Dataset
        Input data.
    sample_dims: Tuple[str]
        Sample dimensions.

    Returns:
    ---------
    feature_dims: Tuple[str]
        Feature dimensions.

    """
    return tuple(dim for dim in data.dims if dim not in sample_dims)


def sqrt_cos_lat_weights(data: DataArray) -> DataArray:
    """Compute the square root of the cosine of the latitude.

    Parameters:
    ------------
    data: xr.DataArray
        Input data.

    Returns:
    ---------
    sqrt_cos_lat: xr.DataArray
        Square root of the cosine of the latitude.

    """
    return xr.apply_ufunc(
        _np_sqrt_cos_lat_weights,
        data,
        vectorize=False,
        dask="allowed",
    )


def total_variance(data: DataArray, dim) -> DataArray:
    """Compute the total variance of the input data.

    Parameters:
    ------------
    data: DataArray
        Input data.

    dim: str
        Dimension along which to compute the total variance.

    Returns:
    ---------
    tot_var: DataArray
        Total variance of the input data.

    """
    return data.var(dim, ddof=1).sum()


def _np_sqrt_cos_lat_weights(data):
    """Compute the square root of the cosine of the latitude.

    Parameters:
    ------------
    data: np.ndarray
        Input data.

    Returns:
    ---------
    sqrt_cos_lat: np.ndarray
        Square root of the cosine of the latitude.

    """
    return np.sqrt(np.cos(np.deg2rad(data)).clip(0, 1))


def get_deterministic_sign_multiplier(data: DataArray, dim: str) -> DataArray:
    """Compute a sign multiplier that ensures deterministic output.

    Uses a method standard to other SVD implementations where we ensure that
    the maximum absolute value in the passed data matrix is positive
    when multipled by the sign multiplier. This creates deterministic output.

    Parameters:
    ------------
    data: DataArray
        Input data to determine sorting order.
    dim: str
        Dimension along which to compute the sign multiplier.

    Returns:
    ---------
    sign_multiplier: DataArray
        Sign multiplier that ensures deterministic output.
    """
    # This method is carefully constructed to avoid idexing ops
    # so we can execute this lazily on dask arrays
    min_max = xr.concat([data.max(dim), data.min(dim)], dim="sign")
    min_max = min_max.assign_coords(sign=[1, -1])
    sign_multiplier = np.abs(min_max).idxmax("sign")
    # Drop all dimensions except 'mode' so that the index is clean
    for dim, coords in sign_multiplier.coords.items():
        if dim != "mode":
            sign_multiplier = sign_multiplier.drop(dim)
    return sign_multiplier


def argsort_dask(data: Data, dim: str) -> Data:
    """Apply argsort to a dask-backed array.

    This is a workaround because dask does not yet implement a chunk-aware
    version of argsort. Therefore we have to force rechunking to a single
    chunk along the sorting dimension and then apply numpy argsort. This
    should be used with an understanding that it may produce large memory
    usage if the passed array is chunked finely along the sorting dimension.

    Parameters:
    ------------
    data: Data
        Input data to sort.
    dim: str
        Dimension along which to sort.

    Returns:
    ---------
    sorted_idx: Data
        Indices that would sort the data.
    """
    return xr.apply_ufunc(
        np.argsort,
        data.chunk({dim: -1}),
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        dask="parallelized",
    )
