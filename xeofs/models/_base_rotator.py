import numpy as np
import scipy as sc
from typing import Optional, Union, List, Tuple

from .eof import EOF
from ..utils.rotation import promax
from ..utils.tools import get_mode_selector


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

    def eofs(self, scaling : int = 0) -> np.ndarray:
        '''EOFs after rotation.

        Parameters
        ----------
        scaling : [0, 1, 2]
            EOFs are scaled (i) to have unit length (``scaling=0``), (ii) by the
            square root of the eigenvalues (``scaling=1``) or (iii) by the
            singular values (``scaling=2``). In case no weights were applied,
            scaling by the singular values results in the EOFs having the
            unit of the input data (the default is 0).

        '''
        dof = self._model.n_samples - 1
        if scaling == 0:
            eofs = self._eofs
        elif scaling == 1:
            eofs = self._eofs * np.sqrt(self._explained_variance)
        elif scaling == 2:
            eofs = self._eofs * np.sqrt(self._explained_variance * dof)
        return eofs

    def pcs(self, scaling : int = 0) -> np.ndarray:
        '''PCs after rotation.

        Parameters
        ----------
        scaling : [0, 1, 2]
            PCs are scaled (i) to have unit length (orthonormal for Varimax
            rotation) (``scaling=0``), (ii) by the square root of the
            eigenvalues (``scaling=1``) or (iii) by the
            singular values (``scaling=2``). In case no weights were applied,
            scaling by the singular values results in the PCs having the
            unit of the input data (the default is 0).

        '''
        dof = self._model.n_samples - 1
        if scaling == 0:
            pcs = self._pcs
        elif scaling == 1:
            pcs = self._pcs * np.sqrt(self._explained_variance)
        elif scaling == 2:
            pcs = self._pcs * np.sqrt(self._explained_variance * dof)
        return pcs

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

    def reconstruct_X(
        self,
        mode : Optional[Union[int, List[int], slice]] = None
    ) -> np.ndarray:
        '''Reconstruct original data field ``X`` using the rotated PCs and EOFs.

        If weights were applied, ``X`` will be automatically rescaled.

        Parameters
        ----------
        mode : Optional[Union[int, List[int], slice]]
            Mode(s) based on which ``X`` will be reconstructed. If ``mode`` is
            an int, a single mode is used. If a list of integers is provided,
            use all specified modes for reconstruction. Alternatively, you may
            want to select a slice to reconstruct. The first mode is denoted
            by 1 (and not by 0). If None then ``X`` is recontructed using all
            available modes (the default is None).

        Examples
        --------

        Perform an analysis using some data ``X``:

        >>> model = EOF(X, norm=True)
        >>> model.solve()

        Reconstruct ``X`` using all modes:

        >>> model.reconstruct_X()

        Reconstruct ``X`` using the first mode only:

        >>> model.reconstruct_X(1)

        Reconstruct ``X`` using mode 1, 3 and 4:

        >>> model.reconstruct_X([1, 3, 4])

        Reconstruct ``X`` using all modes up to mode 10 (including):

        >>> model.reconstruct_X(slice(10))

        Reconstruct ``X`` using every second mode between 4 and 8 (both
        including):

        >>> model.reconstruct_X(slice(4, 8, 2))


        '''
        dof = self._model.n_samples - 1
        eofs = self._eofs
        pcs = self._pcs * np.sqrt(self._explained_variance * dof)
        # Select modes to reconstruct X
        mode = get_mode_selector(mode)
        eofs = eofs[:, mode]
        pcs = pcs[:, mode]
        Xrec = pcs @ eofs.T
        # Unweight and add mean
        return (Xrec / self._model._weights) + self._model._X_mean

    def project_onto_eofs(
        self,
        X : np.ndarray, scaling : int = 0
    ) -> np.ndarray:
        '''Project new data onto the rotated EOFs.

        Parameters
        ----------
        X : np.ndarray
             New data to project. Data must be a 2D matrix.
        scaling : [0, 1, 2]
            Projections are scaled (i) to be orthonormal (``scaling=0``), (ii)
            by the square root of the eigenvalues (``scaling=1``) or (iii) by
            the singular values (``scaling=2``). In the latter case, and when
            no weights were applied, scaling by the singular values results in
            the projections having the unit of the input data
            (the default is 0).

        '''
        dof = self._model.n_samples - 1
        svals = np.sqrt(self._model._explained_variance * dof)

        # Preprocess new data
        try:
            X -= self._model._X_mean
        except ValueError:
            err_msg = (
                'New data has invalid feature dimensions and cannot be '
                'projected onto EOFs.'
            )
            raise ValueError(err_msg)
        X *= self._model._weights

        # Compute non-rotated PCs
        pcs = X @ self._model._eofs[:, :self._n_rot] / svals[:self._n_rot]

        # Rotate and reorder PCs
        R = self._rotation_matrix(inverse_transpose=True)
        pcs = pcs @ R
        pcs = pcs[:, self._idx_var]

        # Apply scaling
        if scaling == 0:
            return pcs
        elif scaling == 1:
            return pcs * np.sqrt(self._explained_variance)
        elif scaling == 2:
            return pcs * np.sqrt(self._explained_variance * dof)
        else:
            err_msg = (
                'Scaling option {:} is not valid but must be one '
                'of [0, 1, 2]'
            )
            err_msg = err_msg.foramt(scaling)
            raise ValueError(err_msg)
