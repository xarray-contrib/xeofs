import numpy as np
import scipy as sc
from typing import Optional, Union, List, Tuple

from ..utils.rotation import promax
from ..utils.tools import get_mode_selector


class _BaseMCARotator:
    '''Rotates a solution obtained from ``xe.models.MCA``.'''

    def __init__(
        self,
        n_rot : int,
        power : int = 1,
        loadings : str = 'standard',
        max_iter : int = 1000,
        rtol : float = 1e-8
    ) -> None:
        '''
        Parameters
        ----------
        model : xe.models.EOF
            A EOF model solution.
        n_rot : int
            Number of modes to be rotated.
        loadings : ['standard', 'squared']
            Rotate the singular vectors either scaled by the square root of
            singular values ('standard') or by the singular values ('squared').
            Using 'standard' loadings, the covariance of the rotated modes will be
            conserved i.e. the sum of the covariance of all rotated modes will
            be equal to the unrotated modes. Using 'squared' loadings, the squared
            covariance will be conserved which then can be used to estimate
            the modes' importance.
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
        self._params = {
            'n_rot': n_rot,
            'power': power,
            'loadings': loadings,
            'max_iter': max_iter,
            'rtol': rtol,
        }
        valid_loadingss = ['standard', 'squared']
        if loadings not in valid_loadingss:
            msg = '{:} is not a valid optione. Muste be one of {:}'
            msg = msg.format(loadings, valid_loadingss)
            raise ValueError(msg)

    def rotate(self, model) -> None:
        n_rot = self._params['n_rot']
        power = self._params['power']
        max_iter = self._params['max_iter']
        rtol = self._params['rtol']
        loadings = self._params['loadings']

        self._model = model

        # Construct combined vector of loadingss
        # NOTE: Dunkerton & Cheng (DC) "load" the combined vectors
        # with the square root of the singular values as it is done for
        # standard Varimax rotation. In this case, the total amount of
        # covariance is conserved under rotation. However, for MCA one
        # typically looks at the squared covariance to investigate the
        # importance of a given mode. Using the approach of DC, the squared
        # covariance is not conserved under rotation thus one cannot estimate
        # the modes' importance after rotation. One way around this issue
        # is to rotate the singular vectors loaded with the singular values
        # ("squared loadingss" as opposed to the square root of the singular
        # values. Like this, the squared covariance is conserved and the
        # importance can be estimated.
        if loadings == 'standard':
            # Original implementation by Dunkerton & Cheng;
            # covariance is conserved, squared covariance is not conserved
            scaling = np.sqrt(model._singular_values[:n_rot])
        elif loadings == 'squared':
            scaling = model._singular_values[:n_rot]

        L = np.concatenate(
            [model._Vx[:, :n_rot], model._Vy[:, :n_rot]],
            axis=0
        )
        L *= scaling

        # Rotate loadingss
        L_rot, rot_mat, corr_mat = promax(
            L, power=power, max_iter=max_iter, rtol=rtol
        )
        # Store rotation and correlation matrix
        self._rot_mat = rot_mat
        self._corr_mat = corr_mat

        # Rotated (loaded) singular vectors for X and Y
        Lx_rot = L_rot[:self._model.n_features_X]
        Ly_rot = L_rot[self._model.n_features_X:]

        # Normalization factor of singular vectors
        self._norm_X = np.linalg.norm(Lx_rot, axis=0)
        self._norm_Y = np.linalg.norm(Ly_rot, axis=0)

        # Rotated (normalized) singular vectors
        self._Vx = Lx_rot / self._norm_X
        self._Vy = Ly_rot / self._norm_Y

        # Remove the squaring introduced from scaling
        if loadings == 'squared':
            self._norm_X = np.sqrt(self._norm_X)
            self._norm_Y = np.sqrt(self._norm_Y)

        # Reorder according to covariance
        self._singular_values = self._norm_X * self._norm_Y
        self._idx_var = np.argsort(self._singular_values)[::-1]

        self._norm_X = self._norm_X[self._idx_var]
        self._norm_Y = self._norm_Y[self._idx_var]
        self._singular_values = self._singular_values[self._idx_var]

        self._Vx = self._Vx[:, self._idx_var]
        self._Vy = self._Vy[:, self._idx_var]

        # Explained covariance and fraction
        self._explained_covariance = self._singular_values
        self._squared_covariance_fraction = np.power(self._singular_values, 2) / self._model._squared_total_variance

        # Rotate PCs using rotatation matrix
        self._Ux = self._model._Ux[:, :n_rot]
        self._Uy = self._model._Uy[:, :n_rot]
        R = self._rotation_matrix(inverse_transpose=True)
        self._Ux = self._Ux @ R
        self._Uy = self._Uy @ R
        # Reorder according to variance
        self._Ux = self._Ux[:, self._idx_var]
        self._Uy = self._Uy[:, self._idx_var]

        # Ensure consistent signs for deterministic output
        maxidx = [abs(L_rot).argmax(axis=0)]
        self._mode_signs = np.sign(L_rot[maxidx, range(L_rot.shape[1])])
        self._Vx *= self._mode_signs
        self._Vy *= self._mode_signs
        self._Ux *= self._mode_signs
        self._Uy *= self._mode_signs

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
        power = self._params['power']

        # only for oblique rotations
        # If rotation is orthogonal: R == R^(-1).T
        # If rotation is oblique (i.e. power>1): R != R^(-1).T
        if inverse_transpose and power > 1:
            R = np.linalg.pinv(R).conjugate().T

        return R

    def singular_values(self) -> np.ndarray:
        '''Get the singular values of the covariance matrix.

        '''
        return self._singular_values

    def explained_covariance(self) -> np.ndarray:
        '''Get the explained covariance.

        The explained covariance is simply given by the individual singular
        values of the covariance matrix.

        '''
        return self._explained_covariance

    def squared_covariance_fraction(self) -> np.ndarray:
        '''Get the squared covariance fraction.

        The squared covariance fraction (SCF) is the fraction of total covariance
        explained by a given mode `i` and is calculated by

        .. math::
           SCF_i = \\frac{\sigma_i^2}{\sum_{i=1}^{m} \sigma_i^2}

        where `m` is the total number of modes and :math:`\sigma_i` denotes the
        `ith` singular value of the covariance matrix.

        '''
        return self._squared_covariance_fraction

    def singular_vectors(
            self, scaling : int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''Get the singular vectors of the covariance matrix.

        Parameters
        ----------
        scaling : [0, 1, 2]
            Singular vectors are scaled (i) to be orthonormal (``scaling=0``),
            (ii) by the square root of the singular values (``scaling=1``) or
            (iii) by the singular values (``scaling=2``) (the default is 0).

        '''
        if scaling == 0:
            Vx = self._Vx
            Vy = self._Vy
        elif scaling == 1:
            Vx = self._Vx * self._norm_X
            Vy = self._Vy * self._norm_Y
        elif scaling == 2:
            Vx = self._Vx * self._norm_X**2
            Vy = self._Vy * self._norm_Y**2
        else:
            msg = (
                'The scaling option {:} is not valid. Please choose one '
                'of the following: [0, 1, 2]'
            )
            msg = msg.format(scaling)
            raise ValueError(msg)
        return Vx, Vy

    def pcs(self, scaling : int = 0) -> Tuple[np.ndarray, np.ndarray]:
        '''Get the PCs.

        The principal components (PCs) in MCA are defined as the projection of
        the data matrices ``X`` and ``Y`` onto the singular vectors.
        There is one set of PCs for each data field.

        Parameters
        ----------
        scaling : [0, 1, 2]
            PCs are scaled (i) to be orthonormal (``scaling=0``), (ii) by the
            square root of the singular values (``scaling=1``) or (iii) by the
            singular values (``scaling=2``) (the default is 0).

        '''
        if scaling == 0:
            Ux = self._Ux
            Uy = self._Uy
        elif scaling == 1:
            Ux = self._Ux * self._norm_X
            Uy = self._Uy * self._norm_Y
        elif scaling == 2:
            Ux = self._Ux * self._norm_X**2
            Uy = self._Uy * self._norm_Y**2
        else:
            msg = (
                'The scaling option {:} is not valid. Please choose one '
                'of the following: [0, 1, 2]'
            )
            msg = msg.format(scaling)
            raise ValueError(msg)
        return Ux, Uy

    def _corr_coef(self, A, B):
        # Columnwise mean of input arrays & subtract from input arrays
        # themeselves
        A_mA = A - A.mean(0)
        B_mB = B - B.mean(0)

        # Sum of squares across rows
        ssA = (A_mA**2).sum(0)
        ssB = (B_mB**2).sum(0)

        # Finally get corr coeff
        return np.dot(A_mA.T, B_mB) / np.sqrt(np.dot(ssA[:, None], ssB[None]))

    def homogeneous_patterns(self) -> Tuple[np.ndarray, np.ndarray]:
        '''Correlation coefficients between PCs and their associated input data.

        More precisely, the homogeneous patterns `r_{hom}` are defined as

        .. math::
          r_{hom, x} = \\corr \\left(X, PC_x \\right)
        .. math::
          r_{hom, y} = \\corr \\left(Y, PC_y \\right)

        Returns
        -------
        Tuple[np.ndarray, np.ndarry]
            Homogeneous patterns for ``X`` and ``Y`` (n_features x n_modes).

        Tuple[np.ndarray, np.ndarry]
             coefficients and associated
            Two-sided p-values for ``X and ``Y`` (n_features x n_modes).
        '''

        # Compute correlation matrix
        corr_X = self._corr_coef(self._model.X, self._Ux)
        corr_Y = self._corr_coef(self._model.Y, self._Uy)

        # Compute two-sided p-values
        # Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html#r8c6348c62346-1
        a = self._model.n_samples / 2 - 1
        dist = sc.stats.beta(a, a, loc=-1, scale=2)
        pvals_X = 2 * dist.cdf(-abs(corr_X))
        pvals_Y = 2 * dist.cdf(-abs(corr_Y))
        return (corr_X, corr_Y), (pvals_X, pvals_Y)

    def heterogeneous_patterns(self) -> Tuple[np.ndarray, np.ndarray]:
        '''Correlation coefficients between PCs and the opposite input data.

        More precisely, the heterogenous patterns `r_{het}` are defined as

        .. math::
          r_{het, x} = \\corr \\left(X, PC_y \\right)
        .. math::
          r_{het, y} = \\corr \\left(Y, PC_x \\right)

        Returns
        -------
        Tuple[np.ndarray, np.ndarry]
            Heterogenous patterns for ``X`` and ``Y`` (n_features x n_modes).

        Tuple[np.ndarray, np.ndarry]
             coefficients and associated
            Two-sided p-values for ``X and ``Y`` (n_features x n_modes).
        '''

        # Compute correlation matrix
        corr_X = self._corr_coef(self._model.X, self._Uy)
        corr_Y = self._corr_coef(self._model.Y, self._Ux)

        # Compute two-sided p-values
        # Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html#r8c6348c62346-1
        a = self._model.n_samples / 2 - 1
        dist = sc.stats.beta(a, a, loc=-1, scale=2)
        pvals_X = 2 * dist.cdf(-abs(corr_X))
        pvals_Y = 2 * dist.cdf(-abs(corr_Y))
        return (corr_X, corr_Y), (pvals_X, pvals_Y)

    def reconstruct_XY(
        self,
        mode : Optional[Union[int, List[int], slice]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''Reconstruct original data fields ``X`` and ``Y``.

        If weights were applied, ``X`` and ``Y`` will be automatically
        rescaled.

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

        >>> model = MCA(X, Y, norm=True)
        >>> model.solve()

        Reconstruct ``X`` and ``Y`` using all modes:

        >>> model.reconstruct_XY()

        Reconstruct ``X`` and ``Y`` using the first mode only:

        >>> model.reconstruct_XY(1)

        Reconstruct ``X`` and ``Y`` using mode 1, 3 and 4:

        >>> model.reconstruct_XY([1, 3, 4])

        Reconstruct ``X`` and ``Y`` using all modes up to mode 10 (including):

        >>> model.reconstruct_XY(slice(10))

        Reconstruct ``X`` and ``Y`` using every second mode between 4 and 8 (both
        including):

        >>> model.reconstruct_XY(slice(4, 8, 2))


        '''
        # Select modes to reconstruct X
        mode = get_mode_selector(mode)

        Vx = self._Vx[:, mode]
        Vy = self._Vy[:, mode]

        Ux = self._Ux[:, mode] * self._norm_X[mode]
        Uy = self._Uy[:, mode] * self._norm_Y[mode]

        Xrec = Ux @ Vx.conj().T
        Yrec = Uy @ Vy.conj().T
        # Unweight and add mean
        Xrec = (Xrec / self._model._weights_X) + self._model._X_mean
        Yrec = (Yrec / self._model._weights_Y) + self._model._Y_mean
        return Xrec, Yrec

    def project_onto_left_singular_vectors(
        self,
        X : np.ndarray = None,
        scaling : int = 0
    ) -> np.ndarray:
        '''Project new data onto the singular vectors.

        Parameters
        ----------
        X : np.ndarray
             New data to project onto first singular vector.
        scaling : [0, 1, 2]
            Projections are scaled (i) to be orthonormal (``scaling=0``), (ii) by the
            square root of the singular values (``scaling=1``) or (iii) by the
            singular values (``scaling=2``) (the default is 0).

        Returns
        -------
        np.ndarray
            Projections of new data onto left singular vector

        '''
        X = X.copy()
        # Remove mean and apply weights
        try:
            X -= self._model._X_mean
        except ValueError:
            err_msg = (
                'New data has invalid feature dimensions and cannot be '
                'projected onto singular vectors. Expected shapes'
            )
            raise ValueError(err_msg)
        X *= self._model._weights_X

        # Project new data onto singular values
        n_rot = self._params['n_rot']
        Ux = X @ self._model._Vx[:, :n_rot] / np.sqrt(self._singular_values)
        # Rotate
        R = self._rotation_matrix(inverse_transpose=True)
        Ux = Ux @ R
        # Reorder
        Ux = Ux[:, self._idx_var]
        # Flip sign
        Ux = Ux * self._mode_signs

        # Apply scaling
        if scaling == 0:
            pass
        elif scaling == 1:
            Ux *= self._norm_X
        elif scaling == 2:
            Ux *= self._norm_X**2
        else:
            err_msg = (
                'Scaling option {:} is not valid but must be one '
                'of [0, 1, 2]'
            )
            err_msg = err_msg.foramt(scaling)
            raise ValueError(err_msg)
        return Ux

    def project_onto_right_singular_vectors(
        self,
        Y : np.ndarray = None,
        scaling : int = 0
    ) -> np.ndarray:
        '''Project new data onto the singular vectors.

        Parameters
        ----------
        Y : np.ndarray
             New data to project onto second singular vector.
        scaling : [0, 1, 2]
            Projections are scaled (i) to be orthonormal (``scaling=0``), (ii) by the
            square root of the singular values (``scaling=1``) or (iii) by the
            singular values (``scaling=2``) (the default is 0).

        Returns
        -------
        np.ndarray
            Projections of new data onto left singular vector

        '''
        Y = Y.copy()
        # Remove mean and apply weights
        try:
            Y -= self._model._Y_mean
        except ValueError:
            err_msg = (
                'New data has invalid feature dimensions and cannot be '
                'projected onto singular vectors. Expected shapes'
            )
            raise ValueError(err_msg)
        Y *= self._model._weights_Y

        # Project new data onto singular values
        n_rot = self._params['n_rot']
        Uy = Y @ self._model._Vy[:, :n_rot] / np.sqrt(self._singular_values)
        # Rotate
        R = self._rotation_matrix(inverse_transpose=True)
        Uy = Uy @ R
        # Reorder
        Uy = Uy[:, self._idx_var]
        # Flip sign
        Uy = Uy * self._mode_signs

        # Apply scaling
        if scaling == 0:
            pass
        elif scaling == 1:
            Uy *= self._norm_Y
        elif scaling == 2:
            Uy *= self._norm_Y**2
        else:
            err_msg = (
                'Scaling option {:} is not valid but must be one '
                'of [0, 1, 2]'
            )
            err_msg = err_msg.foramt(scaling)
            raise ValueError(err_msg)
        return Uy
