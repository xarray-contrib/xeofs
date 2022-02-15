from typing import Iterable

import numpy as np
import pandas as pd

from .. import models
from xeofs.pandas._dataframe_transformer import _DataFrameTransformer


class EOF(models.eof.EOF):

    def __init__(
        self,
        X: Iterable[pd.DataFrame],
        Y: Iterable[pd.DataFrame] = None,
        n_modes=None,
        norm=False
    ):

        if(np.logical_not(isinstance(X, pd.DataFrame))):
            raise ValueError('This interface is for `pandas.DataFrame` only.')

        self._df_tf = _DataFrameTransformer()
        X = self._df_tf.fit_transform(X)

        super().__init__(
            X=X,
            Y=None,
            n_modes=n_modes,
            norm=norm
        )
        self._mode_idx = pd.Index(range(1, self.n_modes + 1), name='mode')

    def singular_values(self):
        svalues = super().singular_values()
        svalues = pd.DataFrame(
            svalues,
            columns=['singular_values'],
            index=self._mode_idx
        )
        return svalues

    def explained_variance(self):
        expvar = super().explained_variance()
        expvar = pd.DataFrame(
            expvar,
            columns=['explained_variance'],
            index=self._mode_idx
        )
        return expvar

    def explained_variance_ratio(self):
        expvar = super().explained_variance_ratio()
        expvar = pd.DataFrame(
            expvar,
            columns=['explained_variance_ratio'],
            index=self._mode_idx
        )
        return expvar

    def eofs(self):
        eofs = self._eofs
        eofs = self._df_tf.back_transform(eofs.T).T
        eofs.columns = self._mode_idx
        return eofs

    def pcs(self):
        pcs = super().pcs()
        return pd.DataFrame(pcs, columns=self._mode_idx, index=self._df_tf.index)
