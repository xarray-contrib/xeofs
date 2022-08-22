from typing import Optional

import numpy as np

from ._base_bootstrapper import _BaseBootstrapper
from ..utils.tools import squeeze


class Bootstrapper(_BaseBootstrapper):
    '''Bootstrapping a numpy EOF model ``xe.models.EOF``.

    '''
    def __init__(
            self, n_boot: int,
            alpha : float = 0.05,
            test_type : Optional[str] = 'one-sided'
    ):
        super().__init__(n_boot=n_boot, alpha=alpha, test_type=test_type)

    def bootstrap(self, model):
        super().bootstrap(model)

    def get_params(self):
        return super().get_params()

    def n_significant_modes(self):
        return super().n_significant_modes()

    def explained_variance(self):
        return super().explained_variance()

    def eofs(self):
        eofs, eofs_mask = super().eofs()
        eofs = [squeeze(self._model._tf.back_transform_eofs(q)) for q in eofs]
        eofs_mask = squeeze(self._model._tf.back_transform_eofs(eofs_mask))
        return eofs, eofs_mask

    def pcs(self):
        pcs, pcs_mask = super().pcs()
        pcs = [squeeze(self._model._tf.back_transform_pcs(q)) for q in pcs]
        pcs_mask = squeeze(self._model._tf.back_transform_pcs(pcs_mask))
        return pcs, pcs_mask
