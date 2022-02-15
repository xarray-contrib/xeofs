import numpy as np
import pandas as pd

from .. import models


class _DataFrameTransformer(models.eof._ArrayTransformer):
    '''`_ArrayTransformer` wrapper for `pandas.DataFrame`.

    '''

    def __init__(self):
        super().__init__()

    def fit(self, data : pd.DataFrame):
        self.index = data.index
        self.columns = data.columns
        super().fit(data.values)
        return self

    def transform(self, data : pd.DataFrame):
        if (data.columns != self.columns).any():
            raise ValueError('Columns are different from fitted data.')
        else:
            return data.values

    def fit_transform(self, data : pd.DataFrame):
        return self.fit(data).transform(data)

    def back_transform(self, data : np.ndarray):
        df = super().back_transform(data)
        return pd.DataFrame(df, columns=self.columns)
