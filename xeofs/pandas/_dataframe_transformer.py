from typing import Union, Iterable
import numpy as np
import pandas as pd

from .. import models


class _DataFrameTransformer(models.eof._ArrayTransformer):
    '''`_ArrayTransformer` wrapper for `pandas.DataFrame`.

    '''

    def __init__(self):
        super().__init__()

    def fit(self, X : pd.DataFrame, axis : Union[int, Iterable[int]] = 0):
        if isinstance(axis, list):
            axis = axis[0]
        # Set sample and feature index
        if axis == 0:
            self.index_samples = X.index
            self.index_features = X.columns
        elif axis == 1:
            self.index_samples = X.columns
            self.index_features = X.index
        else:
            raise ValueError('axis must be either 0 or 1')
        # Fit the data
        try:
            super().fit(X=X.values, axis=axis)
        except AttributeError:
            err_msg = 'weights must be of type {:}.'.format(repr(pd.DataFrame))
            raise TypeError(err_msg)
        return self

    def transform(self, X : pd.DataFrame):
        try:
            return super().transform(X.values)
        except AttributeError:
            err_msg = 'weights must be of type {:}.'.format(repr(pd.DataFrame))
            raise TypeError(err_msg)

    def fit_transform(self, X : pd.DataFrame, axis : int = 0):
        return self.fit(X=X, axis=axis).transform(X)

    def transform_weights(self, weights : pd.DataFrame):
        if weights is None:
            return None
        try:
            return super().transform_weights(weights.values)
        except AttributeError:
            err_msg = 'weights must be of type {:}.'.format(repr(pd.DataFrame))
            raise TypeError(err_msg)

    def back_transform(self, X : np.ndarray):
        df = super().back_transform(X)
        return pd.DataFrame(
            df,
            index=self.index_samples,
            columns=self.index_features
        )

    def back_transform_eofs(self, X : np.ndarray):
        eofs = super().back_transform_eofs(X)
        return pd.DataFrame(
            eofs,
            index=self.index_features,
            columns=range(1, eofs.shape[-1] + 1)
        )

    def back_transform_pcs(self, X : np.ndarray):
        pcs = super().back_transform_pcs(X)
        return pd.DataFrame(
            pcs,
            index=self.index_samples,
            columns=range(1, pcs.shape[-1] + 1)
        )
