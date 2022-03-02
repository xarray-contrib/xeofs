import numpy as np
from typing import Optional, Union, List, Tuple

from .eof import EOF
from ._base_rotator import _BaseRotator
from ._array_transformer import _ArrayTransformer


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

    def explained_variance(self) -> np.ndarray:
        return super().explained_variance()

    def explained_variance_ratio(self) -> np.ndarray:
        return super().explained_variance_ratio()

    def eofs(self, scaling : int = 0) -> np.ndarray:
        eofs = super().eofs(scaling=scaling)
        return self._model._tf.back_transform_eofs(eofs)

    def pcs(self, scaling : int = 0) -> np.ndarray:
        pcs = super().pcs(scaling=scaling)
        return self._model._tf.back_transform_pcs(pcs)

    def eofs_as_correlation(self) -> Tuple[np.ndarray, np.ndarray]:
        corr, pvals = super().eofs_as_correlation()
        corr = self._model._tf.back_transform_eofs(corr)
        pvals = self._model._tf.back_transform_eofs(pvals)
        return corr, pvals

    def reconstruct_X(
        self,
        mode : Optional[Union[int, List[int], slice]] = None
    ) -> np.ndarray:
        Xrec = super().reconstruct_X(mode=mode)
        Xrec = self._model._tf.back_transform(Xrec)
        return Xrec

    def project_onto_eofs(
        self,
        X : np.ndarray,
        scaling : int = 0
    ) -> np.ndarray:
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
        proj = _ArrayTransformer()
        X = proj.fit_transform(X, axis=self._model._tf.axis_samples)
        pcs = super().project_onto_eofs(X=X, scaling=scaling)
        return proj.back_transform_pcs(pcs)
