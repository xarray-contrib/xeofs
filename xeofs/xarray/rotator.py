import xarray as xr

from .eof import EOF
from ..models._base_rotator import _BaseRotator


class Rotator(_BaseRotator):
    '''Rotates a solution obtained from ``xe.xarray.EOF``.

    Parameters
    ----------
    model : xe.xarray.EOF
        A EOF model solution.
    n_rot : int
        Number of modes to be rotated.
    power : int
        Defines the power of Promax rotation. Choosing ``power=1`` equals
        a Varimax solution (the default is 1).
    max_iter : int
        Number of maximal iterations for obtaining the rotation matrix
        (the default is 1000).
    rtol : float
        Relative tolerance to be achieved for early stopping the iteration
        process (the default is 1e-8).


    '''

    def __init__(
        self,
        model : EOF,
        n_rot : int,
        power : int = 1,
        max_iter : int = 1000,
        rtol : float = 1e-8
    ):
        super().__init__(
            model=model, n_rot=n_rot, power=power, max_iter=max_iter, rtol=rtol
        )

    def explained_variance(self) -> xr.DataArray:
        expvar = super().explained_variance()
        return xr.DataArray(
            expvar,
            dims=['mode'],
            coords={'mode' : self._model._idx_mode[:self._n_rot]},
            name='explained_variance'
        )

    def explained_variance_ratio(self) -> xr.DataArray:
        expvar_ratio = super().explained_variance_ratio()
        return xr.DataArray(
            expvar_ratio,
            dims=['mode'],
            coords={'mode' : self._model._idx_mode[:self._n_rot]},
            name='explained_variance_ratio'
        )

    def eofs(self) -> xr.DataArray:
        eofs = super().eofs()
        eofs = self._model._tf.back_transform_eofs(eofs)
        eofs.name = 'EOFs'
        return eofs

    def pcs(self) -> xr.DataArray:
        pcs = super().pcs()
        pcs = self._model._tf.back_transform_pcs(pcs)
        pcs.name = 'PCs'
        return pcs
