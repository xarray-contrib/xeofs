import numpy as np
import pandas as pd
import pytest
import warnings

from xeofs.models._array_transformer import _ArrayTransformer
from xeofs.pandas._dataframe_transformer import _DataFrameTransformer

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


@pytest.mark.parametrize('input_shape', [
    (100, 10),
])
def test_pandas_wrapper(input_shape):
    # Results of Dataframe wrapper and _ArrayTransformer match.
    rng = np.random.default_rng(7)
    arr_in = rng.standard_normal(input_shape)
    df_in = pd.DataFrame(
        arr_in,
        columns=range(1, arr_in.shape[1] + 1),
        index=range(arr_in.shape[0])
    )

    tf1 = _ArrayTransformer()
    tf1.fit(arr_in)
    arr_out = tf1.fit_transform(arr_in)
    arr_back = tf1.back_transform(arr_out)

    tf2 = _DataFrameTransformer()
    tf2.fit(df_in)
    df_out = tf2.fit_transform(df_in)
    df_back = tf2.back_transform(arr_out)

    np.testing.assert_allclose(arr_out, df_out)
    np.testing.assert_allclose(arr_back, df_back.values)
