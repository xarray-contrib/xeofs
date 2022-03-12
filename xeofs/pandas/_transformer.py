from typing import Union, Iterable, List
import numpy as np
import pandas as pd

from ..models._transformer import _ArrayTransformer, _MultiArrayTransformer


class _DataFrameTransformer(_ArrayTransformer):
    '''`_ArrayTransformer` wrapper for `pandas.DataFrame`.

    '''

    def __init__(self):
        super().__init__()

    def fit(self, X : pd.DataFrame, axis : Union[int, Iterable[int]] = 0):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('This interface is for `pandas.DataFrame` only')
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

    def transform(self, X : pd.DataFrame) -> np.ndarray:
        try:
            return super().transform(X.values)
        except AttributeError:
            err_msg = 'weights must be of type {:}.'.format(repr(pd.DataFrame))
            raise TypeError(err_msg)

    def fit_transform(self, X : pd.DataFrame, axis : int = 0) -> np.ndarray:
        return self.fit(X=X, axis=axis).transform(X)

    def transform_weights(self, weights : pd.DataFrame) -> np.ndarray:
        try:
            return super().transform_weights(weights.values)
        except AttributeError:
            return super().transform_weights(weights)

    def back_transform(self, X : np.ndarray) -> pd.DataFrame:
        df = super().back_transform(X)
        return pd.DataFrame(
            df,
            index=self.index_samples,
            columns=self.index_features
        )

    def back_transform_eofs(self, X : np.ndarray) -> pd.DataFrame:
        eofs = super().back_transform_eofs(X)
        return pd.DataFrame(
            eofs,
            index=self.index_features,
            columns=range(1, eofs.shape[-1] + 1)
        )

    def back_transform_pcs(self, X : np.ndarray) -> pd.DataFrame:
        pcs = super().back_transform_pcs(X)
        return pd.DataFrame(
            pcs,
            index=self.index_samples,
            columns=range(1, pcs.shape[-1] + 1)
        )


class _MultiDataFrameTransformer(_MultiArrayTransformer):
    'Transform multiple 2D ``pd.DataFrame`` to a single 2D ``np.ndarry``.'
    def __init__(self):
        super().__init__()

    def fit(self, X : Union[pd.DataFrame, List[pd.DataFrame]], axis : Union[int, Iterable[int]] = 0):
        X = self._convert2list(X)
        self.tfs = [_DataFrameTransformer().fit(x, axis=axis) for x in X]

        if len(set([tf.n_valid_samples for tf in self.tfs])) > 1:
            err_msg = 'All individual arrays must have same number of samples.'
            raise ValueError(err_msg)

        self.idx_array_sep = np.cumsum([tf.n_valid_features for tf in self.tfs])
        self.axis_samples = self.tfs[0].axis_samples
        return self

    def transform(self, X : Union[pd.DataFrame, List[pd.DataFrame]]) -> np.ndarray:
        return super().transform(X=X)

    def transform_weights(self, weights : Union[pd.DataFrame, List[pd.DataFrame]]) -> np.ndarray:
        return super().transform_weights(weights=weights)

    def fit_transform(
        self, X : Union[pd.DataFrame, List[pd.DataFrame]],
        axis : Union[int, Iterable[int]] = 0
    ) -> np.ndarray:
        return self.fit(X=X, axis=axis).transform(X)

    def back_transform(self, X : np.ndarray) -> pd.DataFrame:
        return super().back_transform(X=X)

    def back_transform_eofs(self, X : np.ndarray) -> pd.DataFrame:
        return super().back_transform_eofs(X=X)

    def back_transform_pcs(self, X : np.ndarray) -> pd.DataFrame:
        return super().back_transform_pcs(X=X)
