import numpy as np
import pandas as pd
import pytest
import warnings

from xeofs.models._array_transformer import _ArrayTransformer
from xeofs.pandas._dataframe_transformer import _DataFrameTransformer

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


@pytest.mark.parametrize('shape, axis', [
    ((100, 10), 0),
    ((100, 10), 1),
])
def test_pandas_wrapper(random_array, axis):
    arr_in = random_array
    df_in = pd.DataFrame(arr_in)

    tf1 = _ArrayTransformer()
    tf1.fit(arr_in, axis=axis)
    arr_out = tf1.fit_transform(arr_in)
    arr_back = tf1.back_transform(arr_out)

    tf2 = _DataFrameTransformer()
    tf2.fit(df_in, axis=axis)
    df_out = tf2.fit_transform(df_in)
    df_back = tf2.back_transform(arr_out)

    np.testing.assert_allclose(arr_out, df_out)
    np.testing.assert_allclose(arr_back, df_back.values)


def test_invalid_fit_axis(sample_DataFrame):
    # Exception for axis other than 0 or 1
    tf = _DataFrameTransformer()
    with pytest.raises(Exception):
        tf.fit(sample_DataFrame, axis=2)


@pytest.mark.parametrize('weights', [
    [1, 2, 3],
    4,
])
def test_invalid_data_type(weights, sample_DataFrame):
    # Exception for data type other than pd.DataFrame
    df = sample_DataFrame
    tf = _DataFrameTransformer()
    tf.fit(df)
    with pytest.raises(Exception):
        tf.fit(weights)
    with pytest.raises(Exception):
        tf.transform(weights)
    with pytest.raises(Exception):
        tf.transform_weights(weights)
