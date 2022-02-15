from typing import Iterable, Union

import numpy as np
import xarray as xr

from .. import models
from xeofs.xarray._dataarray_transformer import _DataArrayTransformer


class EOF(models.eof.EOF):

    def __init__(
        self,
        X: Iterable[xr.DataArray],
        Y: Iterable[xr.DataArray] = None,
        n_modes : Union[int, None] = None,
        norm : bool = False,
        dim: str = 'time'
    ):

        if(np.logical_not(isinstance(X, xr.DataArray))):
            raise ValueError('This interface is for `xarray.DataArray` only.')

        self._da_tf = _DataArrayTransformer()
        X = self._da_tf.fit_transform(X, dim=dim)

        super().__init__(
            X=X,
            Y=None,
            n_modes=n_modes,
            norm=norm
        )
        self._mode_idx = xr.IndexVariable('mode', range(1, self.n_modes + 1))
        self._dim = dim

    def singular_values(self):
        svalues = super().singular_values()
        return xr.DataArray(
            svalues,
            dims=['mode'],
            coords={'mode' : self._mode_idx},
            name='singular_values'
        )

    def explained_variance(self):
        expvar = super().explained_variance()
        return xr.DataArray(
            expvar,
            dims=['mode'],
            coords={'mode' : self._mode_idx},
            name='explained_variance'
        )

    def explained_variance_ratio(self):
        expvar = super().explained_variance_ratio()
        return xr.DataArray(
            expvar,
            dims=['mode'],
            coords={'mode' : self._mode_idx},
            name='explained_variance_ratio'
        )

    def eofs(self):
        eofs = self._eofs
        eofs = self._da_tf.back_transform(eofs.T).T
        eofs = eofs.rename({self._dim : 'mode'})
        eofs = eofs.assign_coords({'mode' : self._mode_idx})
        eofs.name = 'EOFs'
        return eofs

    def pcs(self):
        pcs = super().pcs()
        coords = {
            self._dim : self._da_tf.coords_in[self._dim],
            'mode' : self._mode_idx
        }
        return xr.DataArray(
            pcs,
            dims=(list(self._da_tf.dims_sample) + ['mode']),
            coords=coords,
            name='PCs'
        )
