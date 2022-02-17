import numpy as np
import xarray as xr
import pytest
import warnings

from xeofs.models._array_transformer import _ArrayTransformer
from xeofs.xarray._dataarray_transformer import _DataArrayTransformer


warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


@pytest.mark.parametrize('shape, axis', [
    ((100, 10), [0]),
    ((100, 10, 9), [0]),
    ((100, 10, 9), [0, 2]),
])
def test_xarray_wrapper(shape, axis):
    # Results of DataArray wrapper and _ArrayTransformer match.
    rng = np.random.default_rng(7)
    arr_in = rng.standard_normal(shape)
    da_in = xr.DataArray(arr_in)

    tf1 = _ArrayTransformer()
    arr_out = tf1.fit_transform(arr_in, axis=axis)
    arr_back = tf1.back_transform(arr_out)

    dic = {0: 'dim_0', 1: 'dim_1', 2: 'dim_2'}
    dim = [d for a, d in dic.items() if a in axis]
    tf2 = _DataArrayTransformer()
    df_out = tf2.fit_transform(da_in, dim=dim)
    df_back = tf2.back_transform(df_out)

    np.testing.assert_allclose(arr_out, df_out)
    np.testing.assert_allclose(arr_back, df_back.values)


def test_invalid_transform():
    # Dimensions of DataArray to be transformed to not match fitted DataArray.
    rng = np.random.default_rng(7)
    arr_in = rng.standard_normal((100, 3))
    da_in = xr.DataArray(arr_in, dims=['time', 'x'])

    da_new = da_in.copy()
    da_new = da_new.rename({'x' : 'y'})

    tf = _DataArrayTransformer()
    tf.fit(da_in)
    with pytest.raises(Exception):
        _ = tf.transform(da_new)
