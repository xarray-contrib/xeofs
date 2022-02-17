import numpy as np
import pandas as pd

from .. import models


class _DataFrameTransformer(models.eof._ArrayTransformer):
    '''`_ArrayTransformer` wrapper for `pandas.DataFrame`.

    '''

    def __init__(self):
        super().__init__()

    def fit(self, X : pd.DataFrame, axis : int = 0):
        self.index = X.index
        self.columns = X.columns
        super().fit(X=X.values, axis=axis)
        return self

    def transform(self, X : pd.DataFrame):
        return super().transform(X.values)

    def fit_transform(self, X : pd.DataFrame, axis : int = 0):
        return self.fit(X=X, axis=axis).transform(X)

    def back_transform(self, X : np.ndarray):
        df = super().back_transform(X)
        return pd.DataFrame(df, index=self.index, columns=self.columns)

    def back_transform_eofs(self, X : np.ndarray):
        eofs = super().back_transform_eofs(X)
        return pd.DataFrame(eofs, index=self.columns)

    def back_transform_pcs(self, X : np.ndarray):
        pcs = super().back_transform_pcs(X)
        return pd.DataFrame(pcs, index=self.index)
