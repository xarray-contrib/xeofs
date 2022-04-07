import numpy as np
from typing import Optional, Union, List, Tuple

from .eof import EOF
from ._base_rotator import _BaseRotator
from ._transformer import _MultiArrayTransformer
from ..utils.tools import squeeze

Array = np.ndarray
ArrayList = Union[Array, List[Array]]


class Rotator(_BaseRotator):
    '''Rotates a solution obtained from ``xe.models.EOF``.

    Parameters
    ----------
    model : xe.models.EOF
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

    def explained_variance(self) -> Array:
        return super().explained_variance()

    def explained_variance_ratio(self) -> Array:
        return super().explained_variance_ratio()

    def eofs(self, scaling : int = 0) -> ArrayList:
        eofs = super().eofs(scaling=scaling)
        eofs = self._model._tf.back_transform_eofs(eofs)
        return squeeze(eofs)

    def pcs(self, scaling : int = 0) -> Array:
        pcs = super().pcs(scaling=scaling)
        return self._model._tf.back_transform_pcs(pcs)

    def eofs_as_correlation(self) -> Tuple[ArrayList, ArrayList]:
        corr, pvals = super().eofs_as_correlation()
        corr = self._model._tf.back_transform_eofs(corr)
        pvals = self._model._tf.back_transform_eofs(pvals)
        return squeeze(corr), squeeze(pvals)

    def reconstruct_X(
        self,
        mode : Optional[Union[int, List[int], slice]] = None
    ) -> ArrayList:
        Xrec = super().reconstruct_X(mode=mode)
        Xrec = self._model._tf.back_transform(Xrec)
        return squeeze(Xrec)

    def project_onto_eofs(
        self,
        X : ArrayList,
        scaling : int = 0
    ) -> Array:
        '''Project new data onto the rotated EOFs.

        Parameters
        ----------
        X : np.ndarray
             New data to project. Data must have same feature shape as original
             data.
        scaling : [0, 1, 2]
            Projections are scaled (i) to be orthonormal (``scaling=0``), (ii) by the
            square root of the eigenvalues (``scaling=1``) or (iii) by the
            singular values (``scaling=2``). In case no weights were applied,
            scaling by the singular values results in the projections having the
            unit of the input data (the default is 0).

        '''
        proj = _MultiArrayTransformer()
        X = proj.fit_transform(X, axis=self._model._tf.axis_samples)
        pcs = super().project_onto_eofs(X=X, scaling=scaling)
        return proj.back_transform_pcs(pcs)
