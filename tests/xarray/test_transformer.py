import numpy as np
import xarray as xr
import pytest
import warnings

from xeofs.models._transformer import _ArrayTransformer
from xeofs.xarray._transformer import _DataArrayTransformer


warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


@pytest.mark.parametrize('shape, axis, shape_weights, dims_weights', [
    ((100, 10), [0], (10), ('dim_1')),
    ((100, 10, 9), [0], (10, 9), ('dim_1', 'dim_2')),
    ((100, 10, 9), [0, 2], (10), ('dim_1')),
])
def test_xarray_wrapper(random_array, axis, shape_weights, dims_weights):
    # Results of DataArray wrapper and _ArrayTransformer match.
    arr_in = random_array
    arr_weights = np.random.standard_normal(size=shape_weights)
    da_in = xr.DataArray(arr_in)
    da_weights = xr.DataArray(arr_weights, dims=dims_weights)

    tf1 = _ArrayTransformer()
    arr_out = tf1.fit_transform(arr_in, axis=axis)
    arr_weights_out = tf1.transform_weights(arr_weights)
    arr_back = tf1.back_transform(arr_out)

    dic = {0: 'dim_0', 1: 'dim_1', 2: 'dim_2'}
    dim = [d for a, d in dic.items() if a in axis]
    tf2 = _DataArrayTransformer()
    df_out = tf2.fit_transform(da_in, dim=dim)
    da_weights_out = tf2.transform_weights(da_weights)
    df_back = tf2.back_transform(df_out)

    np.testing.assert_allclose(arr_out, df_out)
    np.testing.assert_allclose(arr_weights_out, da_weights_out)
    np.testing.assert_allclose(arr_back, df_back.values)


def test_invalid_transform(sample_DataArray):
    # Dimensions of DataArray to be transformed to not match fitted DataArray.
    da_in = sample_DataArray

    da_new = da_in.copy()
    da_new = da_new.rename({'loc' : 'b'})

    tf = _DataArrayTransformer()
    tf.fit(da_in)
    with pytest.raises(Exception):
        _ = tf.fit(da_in, dim='c')
    with pytest.raises(Exception):
        _ = tf.transform(da_new)
    with pytest.raises(Exception):
        _ = tf.transform_weights(da_new)


@pytest.mark.parametrize('data', [
    np.array([[1, 2], [3, 4]]),
    5
])
def test_invalid_data_types(data, sample_DataArray):
    tf = _DataArrayTransformer()
    tf.fit(sample_DataArray)
    with pytest.raises(Exception):
        _ = tf.transform(data)
    with pytest.raises(Exception):
        _ = tf.transform_weights(data)
