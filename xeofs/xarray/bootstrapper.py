from typing import Optional

import xarray as xr

from ..models._base_bootstrapper import _BaseBootstrapper
from ..utils.tools import squeeze


class Bootstrapper(_BaseBootstrapper):
    '''Bootstrapping an xarray EOF model ``xe.xarray.EOF``.

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
        expvar, expvar_mask = super().explained_variance()
        expvar = xr.DataArray(
            expvar,
            dims=['quantile', 'mode'],
            coords=dict(
                quantile=self._params['quantiles'],
                mode=self._model._idx_mode
            ),
            name='explained_variance_uncertainty'
        )
        expvar_mask = xr.DataArray(
            expvar_mask,
            dims=['mode'],
            coords=dict(
                mode=self._model._idx_mode[:-1]
            ),
            name='mode_is_significant'
        )
        return expvar, expvar_mask

    def eofs(self):
        eofs, eofs_mask = super().eofs()
        # Merge quantiles into one DataArray
        qlow, qup = [self._model._tf.back_transform_eofs(eof) for eof in eofs]
        eofs = [
            xr.concat([q0, q1], dim='quantile').assign_coords(dict(quantile=self._params['quantiles']))
            for q0, q1 in zip(qlow, qup)
        ]
        eofs_mask = self._model._tf.back_transform_eofs(eofs_mask)

        # Floats are returned, convert to boolean mask
        eofs_mask = [eof.astype(bool) for eof in eofs_mask]
        eofs_mask = squeeze(eofs_mask)

        for i in range(len(eofs)):
            eofs[i].name = 'eofs_uncertainty'
            eofs_mask[i].name = 'eof_is_significant'

        return squeeze(eofs), squeeze(eofs_mask)

    def pcs(self):
        pcs, pcs_mask = super().pcs()
        # Merge quantiles into one DataArray
        qlow, qup = [self._model._tf.back_transform_pcs(pc) for pc in pcs]
        pcs = xr.concat([qlow, qup], dim='quantile').assign_coords(dict(quantile=self._params['quantiles']))
        pcs_mask = self._model._tf.back_transform_pcs(pcs_mask)

        # Floats are returned, convert to boolean mask
        pcs_mask = pcs_mask.astype(bool)

        pcs.name = 'pcs_uncertainty'
        pcs_mask.name = 'pc_is_significant'

        return squeeze(pcs), squeeze(pcs_mask)
