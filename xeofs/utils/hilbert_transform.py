import numpy as np
import xarray as xr
from scipy.signal import hilbert  # type: ignore
from .data_types import DataArray


def hilbert_transform(
    data: DataArray, dims, padding: str = "exp", decay_factor: float = 0.2
) -> DataArray:
    """Hilbert transform with optional padding to mitigate spectral leakage.

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

    """
    return xr.apply_ufunc(
        _hilbert_transform_with_padding,
        data,
        input_core_dims=[dims],
        output_core_dims=[dims],
        kwargs={"padding": padding, "decay_factor": decay_factor},
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
    )


def _hilbert_transform_with_padding(y, padding: str = "exp", decay_factor: float = 0.2):
    """Hilbert transform with optional padding to mitigate spectral leakage.

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

    """
    n_samples = y.shape[0]

    if padding == "exp":
        y = _pad_exp(y, decay_factor=decay_factor)

    y = hilbert(y, axis=0)

    if padding == "exp":
        y = y[n_samples : 2 * n_samples]

    # Padding can introduce a shift in the mean of the imaginary part
    # of the Hilbert transform. Correct for this shift.
    y = y - y.mean(axis=0)  # type: ignore

    return y


def _pad_exp(y, decay_factor: float = 0.2):
    """Pad the input array with an exponential decay function.

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

    """
    x = np.arange(y.shape[0])
    x_ext = np.arange(-x.size, 2 * x.size)

    coefs = np.polynomial.polynomial.polyfit(x, y, deg=1)
    yfit = np.polynomial.polynomial.polyval(x, coefs).T
    yfit_ext = np.polynomial.polynomial.polyval(x_ext, coefs).T

    y_ano = y - yfit

    amp_pre = np.take(y_ano, 0, axis=0)[:, None]
    amp_pos = np.take(y_ano, -1, axis=0)[:, None]

    exp_ext = np.exp(-x / x.size / decay_factor)
    exp_ext_reverse = exp_ext[::-1]

    pad_pre = amp_pre * exp_ext_reverse
    pad_pos = amp_pos * exp_ext

    y_ext = np.concatenate([pad_pre.T, y_ano, pad_pos.T], axis=0)
    y_ext += yfit_ext
    return y_ext
