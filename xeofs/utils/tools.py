from typing import Optional, Union, List, Sequence, Hashable

import numpy as np
import xarray as xr
from scipy.signal import hilbert    # type: ignore

from .sanity_checks import ensure_tuple
from .data_types import XarrayData, DataArrayList, DataArray, Dataset

def get_dims(
        data: DataArray | Dataset | List[DataArray],
        sample_dims: Hashable | Sequence[Hashable] | List[Sequence[Hashable]]
        ):

    # Check for invalid types
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        sample_dims = ensure_tuple(sample_dims)
        feature_dims = _get_complementary_dims(data, sample_dims)

    elif isinstance(data, list):
        sample_dims = ensure_tuple(sample_dims)
        feature_dims = [_get_complementary_dims(da, sample_dims) for da in data]
    else:
        err_message = f'Invalid input type: {type(data).__name__}. Expected one of '
        err_message += f'of the following: DataArray, Dataset or list of DataArrays.'
        raise TypeError(err_message)

    return sample_dims, feature_dims

def _get_complementary_dims(data, sample_dims):
    feature_dims = tuple([dim for dim in data.dims if dim not in sample_dims])
    return feature_dims


def get_mode_selector(obj : Optional[Union[int, List[int], slice]]) -> Union[slice, List]:
    ''' Create a mode selector for a given input object.

    Lists are returned as lists. All other possible input types
    are returned as slices.

    Parameters
    ----------
    obj : Optional[Union[int, List[int], slice]]
        Data type to be casted as a mode selector.


    '''
    MAX_MODE = 9999999
    if obj is None:
        return slice(MAX_MODE)
    elif isinstance(obj, int):
        return [obj - 1]
    elif isinstance(obj, slice):
        # Reduce slice start by one so that "1" is the first element
        try:
            new_start = obj.start - 1
        except TypeError:
            new_start = 0
        # Slice start cannot be negative
        new_start = max(0, new_start)
        return slice(new_start, obj.stop, obj.step)
    elif isinstance(obj, list):
        # Reduce all list elements by 1 so that "1" is first element
        return [o - 1 for o in obj]
    else:
        obj_type = type(obj)
        err_msg = 'Invalid type {:}. Must be one of [int, slice, list, None].'
        err_msg = err_msg.format(obj_type)
        raise ValueError(err_msg)


def squeeze(ls):
    '''Squeeze a list.

    If list is of length 1 return the element, otherwise return the list.
    '''
    if len(ls) > 1:
        return ls
    elif len(ls) == 1:
        return ls[0]
    else:
        raise IndexError('list is empty')


def np_sqrt_cos_lat_weights(arr):
    return np.sqrt(np.cos(np.deg2rad(arr))).clip(0, 1)

def np_total_variance(arr):
    C = (arr * arr.conj()).sum(axis=0) / (arr.shape[0] - 1)
    return C.sum().real

def compute_total_variance(data):
    tot_var = xr.apply_ufunc(
        np_total_variance,
        data,
        input_core_dims=[['sample', 'feature']],
        output_core_dims=[[]],
        vectorize=False,
        dask='allowed',
        output_dtypes=[float],
    )
    tot_var.name = 'total_variance'
    return tot_var

def _hilbert_transform_with_padding(y, decay_factor=.2):
    n_samples = y.shape[0]

    y = _pad_exp(y, decay_factor=decay_factor)
    y = hilbert(y, axis=0)
    y = y[n_samples:2*n_samples]
    return y

def _pad_exp(y, decay_factor=.2):
    x = np.arange(y.shape[0])
    x_ext = np.arange(-x.size, 2*x.size)

    coefs = np.polynomial.polynomial.polyfit(x, y, deg=1)
    yfit = np.polynomial.polynomial.polyval(x, coefs).T
    yfit_ext= np.polynomial.polynomial.polyval(x_ext, coefs).T

    y_ano = y - yfit

    amp_pre = y_ano.take(0, axis=0)[:,None]
    amp_pos = y_ano.take(-1, axis=0)[:,None]

    exp_ext = np.exp(-x / x.size / decay_factor)
    exp_ext_reverse = exp_ext[::-1]
    
    pad_pre = amp_pre * exp_ext_reverse
    pad_pos = amp_pos * exp_ext

    y_ext = np.concatenate([pad_pre.T, y_ano, pad_pos.T], axis=0)
    y_ext += yfit_ext
    return y_ext
