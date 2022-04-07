import xarray as xr
from typing import Tuple, Optional, Union, List

from .eof import EOF
from ..models._base_rotator import _BaseRotator
from ..utils.tools import squeeze
from ._transformer import _MultiDataArrayTransformer

DataArray = xr.DataArray
DataArrayList = Union[DataArray, List[DataArray]]


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

    def explained_variance(self) -> DataArray:
        expvar = super().explained_variance()
        return xr.DataArray(
            expvar,
            dims=['mode'],
            coords={'mode' : self._model._idx_mode[:self._n_rot]},
            name='explained_variance'
        )

    def explained_variance_ratio(self) -> DataArray:
        expvar_ratio = super().explained_variance_ratio()
        return xr.DataArray(
            expvar_ratio,
            dims=['mode'],
            coords={'mode' : self._model._idx_mode[:self._n_rot]},
            name='explained_variance_ratio'
        )

    def eofs(self, scaling : int = 0) -> DataArrayList:
        eofs = super().eofs(scaling=scaling)
        eofs = self._model._tf.back_transform_eofs(eofs)
        return squeeze(eofs)

    def pcs(self, scaling : int = 0) -> DataArray:
        pcs = super().pcs(scaling=scaling)
        pcs = self._model._tf.back_transform_pcs(pcs)
        return pcs

    def eofs_as_correlation(self) -> Tuple[DataArrayList, DataArrayList]:
        corr, pvals = super().eofs_as_correlation()
        corr = self._model._tf.back_transform_eofs(corr)
        pvals = self._model._tf.back_transform_eofs(pvals)
        for c, p in zip(corr, pvals):
            c.name = 'correlation_coeffient'
            p.name = 'p_value'
        return squeeze(corr), squeeze(pvals)

    def reconstruct_X(
        self,
        mode : Optional[Union[int, List[int], slice]] = None
    ) -> DataArray:
        Xrec = super().reconstruct_X(mode=mode)
        Xrec = self._model._tf.back_transform(Xrec)
        return squeeze(Xrec)

    def project_onto_eofs(
        self,
        X : DataArrayList,
        scaling : int = 0
    ) -> DataArray:
        '''Project new data onto the rotated EOFs.

        Parameters
        ----------
        X : xr.DataArray
             New data to project. Data must have same feature shape as original
             data.
        scaling : [0, 1, 2]
            Projections are scaled (i) to be orthonormal (``scaling=0``), (ii) by the
            square root of the eigenvalues (``scaling=1``) or (iii) by the
            singular values (``scaling=2``). In case no weights were applied,
            scaling by the singular values results in the projections having the
            unit of the input data (the default is 0).

        '''
        proj = _MultiDataArrayTransformer()
        X = proj.fit_transform(X, dim=self._model._tf.dims_samples)
        pcs = super().project_onto_eofs(X=X, scaling=scaling)
        return proj.back_transform_pcs(pcs)
