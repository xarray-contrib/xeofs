from typing import List, Sequence, Hashable, Tuple

import numpy as np
import xarray as xr
from scipy.signal import hilbert    # type: ignore

from .sanity_checks import ensure_tuple
from .data_types import XarrayData, DataArray, Dataset, SingleDataObject
from .constants import VALID_LATITUDE_NAMES


def compute_sqrt_cos_lat_weights(data: SingleDataObject, dim: Hashable | Sequence[Hashable]) -> SingleDataObject:
        '''Compute the square root of cosine of latitude weights.

        Parameters
        ----------
        data : xarray.DataArray or xarray.Dataset
            Data to be scaled.
        dim : sequence of hashable 
            Dimensions along which the data is considered to be a feature.
            
        Returns
        -------
        xarray.DataArray or xarray.Dataset
            Square root of cosine of latitude weights.

        '''
        dim = ensure_tuple(dim)

        # Find latitude coordinate
        is_lat_coord = np.isin(np.array(dim), VALID_LATITUDE_NAMES)

        # Select latitude coordinate and compute coslat weights
        lat_coord = np.array(dim)[is_lat_coord]
        
        if len(lat_coord) > 1:
            raise ValueError(f'{lat_coord} are ambiguous latitude coordinates. Only ONE of the following is allowed for computing coslat weights: {VALID_LATITUDE_NAMES}')

        if len(lat_coord) == 1:
            latitudes = data.coords[lat_coord[0]]
            weights = sqrt_cos_lat_weights(latitudes)
            # Features that cannot be associated to a latitude receive a weight of 1
            weights = weights.where(weights.notnull(), 1)
        else:
            raise ValueError('No latitude coordinate was found to compute coslat weights. Must be one of the following: {:}'.format(VALID_LATITUDE_NAMES))
        weights.name = 'coslat_weights'
        return weights


def get_dims(
        data: DataArray | Dataset | List[DataArray],
        sample_dims: Hashable | Sequence[Hashable] | List[Sequence[Hashable]]
        ) -> Tuple[Hashable, Hashable]:
    '''Extracts the dimensions of a DataArray or Dataset that are not included in the sample dimensions.

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

    '''
    # Check for invalid types
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        sample_dims = ensure_tuple(sample_dims)
        feature_dims = _get_feature_dims(data, sample_dims)

    elif isinstance(data, list):
        sample_dims = ensure_tuple(sample_dims)
        feature_dims = [_get_feature_dims(da, sample_dims) for da in data]
    else:
        err_message = f'Invalid input type: {type(data).__name__}. Expected one of '
        err_message += f'of the following: DataArray, Dataset or list of DataArrays.'
        raise TypeError(err_message)

    return sample_dims, feature_dims  # type: ignore

def _get_feature_dims(data: XarrayData, sample_dims: Tuple[str]) -> Tuple[Hashable]:
    '''Extracts the dimensions of a DataArray that are not included in the sample dimensions.


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

    '''
    feature_dims = tuple(dim for dim in data.dims if dim not in sample_dims)
    return feature_dims


def sqrt_cos_lat_weights(data: SingleDataObject) -> SingleDataObject:
    '''Compute the square root of the cosine of the latitude.

    Parameters:
    ------------
    data: xr.DataArray or xr.Dataset
        Input data.

    Returns:
    ---------
    sqrt_cos_lat: xr.DataArray or xr.Dataset
        Square root of the cosine of the latitude.

    '''
    return xr.apply_ufunc(
        _np_sqrt_cos_lat_weights,
        data,
        vectorize=False,
        dask='allowed',
    )


def total_variance(data: DataArray, dim) -> DataArray:
    '''Compute the total variance of the input data.
    
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

    '''
    return data.var(dim, ddof=1).sum()

def hilbert_transform(data: DataArray, dim, padding='exp', decay_factor=.2) -> DataArray:
    '''Hilbert transform with optional padding to mitigate spectral leakage.

    Parameters:
    ------------
    data: DataArray
        Input data.
    dim: str
        Dimension along which to apply the Hilbert transform.
    padding: str
        Padding type. Can be 'exp' or None.
    decay_factor: float
        Decay factor of the exponential function.

    Returns:
    ---------
    data: DataArray
        Hilbert transform of the input data.

    '''
    return xr.apply_ufunc(
        _hilbert_transform_with_padding,
        data,
        input_core_dims=[['sample', 'feature']],
        output_core_dims=[['sample', 'feature']],
        kwargs={'padding': padding, 'decay_factor': decay_factor},
        dask='parallelized',
        dask_gufunc_kwargs={'allow_rechunk': True}
    )

def _np_sqrt_cos_lat_weights(data):
    '''Compute the square root of the cosine of the latitude.
    
    Parameters:
    ------------
    data: np.ndarray
        Input data.

    Returns:
    ---------
    sqrt_cos_lat: np.ndarray
        Square root of the cosine of the latitude.

    '''
    return np.sqrt(np.cos(np.deg2rad(data))).clip(0, 1)


def _hilbert_transform_with_padding(y, padding='exp', decay_factor=.2):
    '''Hilbert transform with optional padding to mitigate spectral leakage.
    
    Parameters:
    ------------
    y: np.ndarray
        Input array.
    padding: str
        Padding type. Can be 'exp' or None.
    decay_factor: float
        Decay factor of the exponential function.

    Returns:
    ---------
    y: np.ndarray
        Hilbert transform of the input array.

    '''
    n_samples = y.shape[0]

    if padding == 'exp':
        y = _pad_exp(y, decay_factor=decay_factor)
    
    y = hilbert(y, axis=0)
    
    if padding == 'exp':
        y = y[n_samples:2*n_samples]

    # Padding can introduce a shift in the mean of the imaginary part
    # of the Hilbert transform. Correct for this shift.
    y = y - y.mean(axis=0)

    return y

def _pad_exp(y, decay_factor=.2):
    '''Pad the input array with an exponential decay function.

    The start and end of the input array are padded with an exponential decay
    function falling to a reference line given by a linear fit of the data array.
    
    Parameters:
    ------------
    y: np.ndarray
        Input array.
    decay_factor: float
        Decay factor of the exponential function.
        
    Returns:
    ---------
    y_ext: np.ndarray
        Padded array.
        
    '''
    x = np.arange(y.shape[0])
    x_ext = np.arange(-x.size, 2*x.size)

    coefs = np.polynomial.polynomial.polyfit(x, y, deg=1)
    yfit = np.polynomial.polynomial.polyval(x, coefs).T
    yfit_ext= np.polynomial.polynomial.polyval(x_ext, coefs).T

    y_ano = y - yfit

    amp_pre = np.take(y_ano, 0, axis=0)[:,None]
    amp_pos = np.take(y_ano, -1, axis=0)[:,None]

    exp_ext = np.exp(-x / x.size / decay_factor)
    exp_ext_reverse = exp_ext[::-1]
    
    pad_pre = amp_pre * exp_ext_reverse
    pad_pos = amp_pos * exp_ext

    y_ext = np.concatenate([pad_pre.T, y_ano, pad_pos.T], axis=0)
    y_ext += yfit_ext
    return y_ext
