from typing import Optional

import numpy as np
import pandas as pd

from ..models._base_bootstrapper import _BaseBootstrapper
from .eof import EOF
from ..utils.tools import squeeze


class Bootstrapper(_BaseBootstrapper):
    '''Bootstrapping a pandas EOF model ``xe.pandas.EOF``.

    '''
    def __init__(
            self, n_boot : int,
            alpha : float = 0.05,
            test_type : Optional[str] = 'one-sided'
    ):
        super().__init__(n_boot=n_boot, alpha=alpha, test_type=test_type)

    def bootstrap(self, model : EOF):
        super().bootstrap(model)

    def get_params(self):
        return super().get_params()

    def n_significant_modes(self):
        return super().n_significant_modes()

    def explained_variance(self):
        expvar, expvar_mask = super().explained_variance()
        expvar = pd.DataFrame(
            expvar.T,
            columns=pd.Index(self._params['quantiles'], name='quantile'),
            index=self._model._idx_mode,
        )
        expvar_mask = pd.DataFrame(
            expvar_mask,
            columns=['is_significant'],
            index=self._model._idx_mode[:-1]
        )

        return expvar, expvar_mask

    def eofs(self):
        eofs, eofs_mask = super().eofs()
        eofs = [squeeze(self._model._tf.back_transform_eofs(e)) for e in eofs]
        eofs_mask = squeeze(self._model._tf.back_transform_eofs(eofs_mask))

        return eofs, eofs_mask

    def pcs(self):
        pcs, pcs_mask = super().pcs()
        pcs = [squeeze(self._model._tf.back_transform_pcs(q)) for q in pcs]
        pcs_mask = squeeze(self._model._tf.back_transform_pcs(pcs_mask))
        return pcs, pcs_mask
