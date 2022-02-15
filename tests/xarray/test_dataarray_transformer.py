import numpy as np
import xarray as xr
import pytest

from xeofs.models._array_transformer import _ArrayTransformer
from xeofs.xarray._dataarray_transformer import _DataArrayTransformer


@pytest.mark.parametrize('input_shape', [
    (100, 10),
])
def test_pandas_wrapper(input_shape):
    # Results of DataArray wrapper and _ArrayTransformer match.
    rng = np.random.default_rng(7)
    arr_in = rng.standard_normal(input_shape)
    da_in = xr.DataArray(arr_in)

    tf1 = _ArrayTransformer()
    tf1.fit(arr_in)
    arr_out = tf1.fit_transform(arr_in)
    arr_back = tf1.back_transform(arr_out)

    tf2 = _DataArrayTransformer()
    tf2.fit(da_in, dim='dim_0')
    df_out = tf2.fit_transform(da_in, dim='dim_0')
    df_back = tf2.back_transform(arr_out)

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
