from typing import Sequence, Hashable, Tuple

import numpy as np
import xarray as xr

from .sanity_checks import convert_to_dim_type
from .data_types import (
    Dims,
    DimsList,
    DataArray,
    DataSet,
    DataList,
)
from .constants import VALID_LATITUDE_NAMES


def compute_sqrt_cos_lat_weights(
    data: DataArray | DataSet, dim: Hashable | Sequence[Hashable]
) -> DataArray:
    """Compute the square root of cosine of latitude weights.

    Parameters
    ----------
    data : xr.DataArray
        Data to be scaled.
    dim : sequence of hashable
        Dimensions along which the data is considered to be a feature.

    Returns
    -------
    xr.DataArray
        Square root of cosine of latitude weights.

    """

    if isinstance(data, (xr.DataArray, xr.Dataset)):
        dim = convert_to_dim_type(dim)
        # Find latitude coordinate
        is_lat_coord = np.isin(np.array(dim), VALID_LATITUDE_NAMES)

        # Select latitude coordinate and compute coslat weights
        lat_coord = np.array(dim)[is_lat_coord]

        if len(lat_coord) > 1:
            raise ValueError(
                f"{lat_coord} are ambiguous latitude coordinates. Only ONE of the following is allowed for computing coslat weights: {VALID_LATITUDE_NAMES}"
            )

        if len(lat_coord) == 1:
            latitudes: DataArray = data.coords[lat_coord[0]]
            assert isinstance(latitudes, xr.DataArray)
            weights = sqrt_cos_lat_weights(latitudes)
            # Features that cannot be associated to a latitude receive a weight of 1
            weights = weights.where(weights.notnull(), 1)
        else:
            raise ValueError(
                "No latitude coordinate was found to compute coslat weights. Must be one of the following: {:}".format(
                    VALID_LATITUDE_NAMES
                )
            )
        weights.name = "coslat_weights"
        return weights

    else:
        raise TypeError(
            "Invalid input type: {:}. Expected one of the following: DataArray".format(
                type(data).__name__
            )
        )


def get_dims(
    data: DataArray | DataSet | DataList,
    sample_dims: Hashable | Sequence[Hashable],
) -> Tuple[Dims, Dims | DimsList]:
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
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        sample_dims = convert_to_dim_type(sample_dims)
        feature_dims: Dims = _get_feature_dims(data, sample_dims)
        return sample_dims, feature_dims

    elif isinstance(data, list):
        sample_dims = convert_to_dim_type(sample_dims)
        feature_dims: DimsList = [_get_feature_dims(da, sample_dims) for da in data]
        return sample_dims, feature_dims
    else:
        err_message = f"Invalid input type: {type(data).__name__}. Expected one of "
        err_message += f"of the following: DataArray, Dataset or list of DataArrays."
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
