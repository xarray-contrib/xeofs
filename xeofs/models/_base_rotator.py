import numpy as np
import scipy as sc
from typing import Tuple

from .eof import EOF
from ..utils.rotation import promax


class _BaseRotator:
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
        self._power = power
        self._n_rot = n_rot
        self._model = model
        eofs = model._eofs[:, :n_rot]
        eigenvalues = model._explained_variance[:n_rot]
        L = eofs * np.sqrt(eigenvalues)
        eofs_rot, rot_mat, corr_mat = promax(
            L, power=power, max_iter=max_iter, rtol=rtol
        )
        self._rot_mat = rot_mat
        self._explained_variance = (eofs_rot ** 2).sum(axis=0)
        self._idx_var = np.argsort(self._explained_variance)[::-1]
        # Reorder rotated EOFs according to their variance
        self._eofs = eofs_rot[:, self._idx_var]
        self._explained_variance = self._explained_variance[self._idx_var]
        # Calculate explained variance fraction by using total variance
        expvar = self._model._explained_variance[0]
        frac_expvar = self._model._explained_variance_ratio[0]
        total_variance = expvar / frac_expvar
        # Calculate variance ratio
        self._explained_variance_ratio = self._explained_variance / total_variance

        # Normalize EOFs
        self._eofs = self._eofs / np.sqrt(self._explained_variance)

        # Rotate PCs using rotatation matrix
        self._pcs = self._model._pcs[:, :self._n_rot]
        R = self._rotation_matrix(inverse_transpose=True)
        self._pcs = self._pcs @ R
        # Reorder according to variance
        self._pcs = self._pcs[:, self._idx_var]

    def _rotation_matrix(self, inverse_transpose : bool = False) -> np.ndarray:
        '''Return the rotation matrix.

        Parameters
        ----------
        inverse_transpose : boolean
            If True, return the inverse transposed of the rotation matrix.
            For orthogonal rotations (Varimax) the inverse transpose equals
            the rotation matrix itself. For oblique rotations (Promax), it
            will be different in general (the default is False).

        Returns
        -------
        rotation_matrix : np.ndarray

        '''
        R = self._rot_mat

        # only for oblique rotations
        # If rotation is orthogonal: R == R^(-1).T
        # If rotation is oblique (i.e. power>1): R != R^(-1).T
        if inverse_transpose and self._power > 1:
            R = np.linalg.pinv(R).conjugate().T

        return R

    def explained_variance(self) -> np.ndarray:
        '''Explained variance after rotation.'''

        return self._explained_variance

    def explained_variance_ratio(self) -> np.ndarray:
        '''Explained variance ratio after rotation.'''

        return self._explained_variance_ratio

    def eofs(self) -> np.ndarray:
        '''EOFs after rotation.'''

        return self._eofs

    def pcs(self) -> np.ndarray:
        '''PCs after rotation.'''

        return self._pcs

    def eofs_as_correlation(self) -> Tuple[np.ndarray, np.ndarray]:
        '''Correlation coefficients between rotated PCs and data matrix.

        Returns
        -------
        Tuple[np.ndarray, np.ndarry]
            Matrices of correlation coefficients and associated
            two-sided p-values with features as rows and modes as columns.

        '''

        # Compute correlation matrix
        corr = np.corrcoef(self._model.X, self._pcs, rowvar=False)
        corr = corr[:self._model.n_features, self._model.n_features:]
        # Compute two-sided p-values
        # Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html#r8c6348c62346-1
        a = self._model.n_samples / 2 - 1
        dist = sc.stats.beta(a, a, loc=-1, scale=2)
        pvals = 2 * dist.cdf(-abs(corr))
        return corr, pvals
