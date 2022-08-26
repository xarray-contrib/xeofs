from typing import Optional, Tuple, List, Union

import numpy as np
import scipy as sc
from sklearn.utils.extmath import randomized_svd

from ..utils.tools import get_mode_selector


class _BaseMCA():
    '''Base class for Maximum Covariance Analysis (MCA).

    Parameters
    ----------
    X : np.ndarray
        2D data matrix ``X`` with samples as rows and features as columns.
        NaN entries will raise an error.
    Y : np.ndarray
        2D data matrix ``Y`` with samples as rows and features as columns.
        NaN entries will raise an error.
    n_modes : Optional[int]
        Number of modes to be retrieved. If None, then all possible modes will
        be computed. Reducing ``n_modes`` can greatly speed up computational
        (the default is None).
    norm : bool
        Normalize each feature by its standard deviation (the default is False).
    weights_X : Optional[np.ndarray]
        Apply `weights` to features of ``X``. Weights must be a 1D array with the same
        length as number of features. No NaN entries are allowed
        (the default is None).
    weights_y : Optional[np.ndarray]
        Apply `weights_Y` to features of ``Y``. Weights must be a 1D array with the same
        length as number of features. No NaN entries are allowed
        (the default is None).

    Attributes
    ----------
    n_samples : int
        Number of samples (rows of data matrix).
    n_features : int
        Number of features (columns of data matrix).
    n_modes : int
        Number of modes.

    '''

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_modes: Optional[int] = None,
        norm: bool = False,
        weights_X: Optional[np.ndarray] = None,
        weights_Y: Optional[np.ndarray] = None
    ):
        # Remove mean for each feature
        self._X_mean = X.mean(axis=0)
        self._Y_mean = Y.mean(axis=0)
        X -= self._X_mean
        Y -= self._Y_mean

        # Weights are applied to features, not samples.
        self._weights_X = weights_X
        self._weights_Y = weights_Y
        # Use int type to ensure that there won't be rounding errors
        # when applying trivial weighting (= all weights equal 1)
        if self._weights_X is None:
            self._weights_X = np.ones(X.shape[1], dtype=int)
        if self._weights_Y is None:
            self._weights_Y = np.ones(Y.shape[1], dtype=int)

        # Standardization is included as weights
        if norm:
            stdev_X = X.std(axis=0)
            stdev_Y = Y.std(axis=0)
            eps = 1e-8
            if (stdev_X < eps).any() or (stdev_Y < eps).any():
                err_msg = (
                    'Standard deviation of one ore more features is zero, '
                    'normalization not possible.'
                )
                raise ValueError(err_msg)
            self._weights_X = self._weights_X / stdev_X
            self._weights_Y = self._weights_Y / stdev_Y
        X = X * self._weights_X
        Y = Y * self._weights_Y

        n_samples_X = X.shape[0]
        n_samples_Y = Y.shape[0]
        if n_samples_X != n_samples_Y:
            msg = (
                'X and Y must have same number of samples but '
                'got {:} and {:}'
            )
            msg = msg.format(n_samples_X, n_samples_Y)
            raise ValueError(msg)

        self.n_samples = X.shape[0]
        self.n_features_X = X.shape[1]
        self.n_features_Y = Y.shape[1]

        self.n_modes = n_modes
        if n_modes is None:
            self.n_modes = min(self.n_features_X - 1, self.n_features_Y - 1)

        self.X = X
        self.Y = Y

    def solve(self) -> None:
        '''
        Perform MCA (aka SVD analysis).

        To boost performance, the standard solver is based on
        the truncated randomized SVD implementation of scikit-learn [1]_ .


        References
        ----------
        .. [1] https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html

        '''
        C = self.X.T @ self.Y / (self.n_samples - 1)
        self._squared_total_variance = np.power(C, 2).sum()

        Vx, svalues, Vy = randomized_svd(
            C, n_components=self.n_modes, n_iter=5, random_state=None
        )
        Vy = Vy.T
        self._Vx = Vx
        self._Vy = Vy
        # Note:
        # - explained variance is given by the singular values of the SVD;
        # - We use the term singular_values_X as used in the context of PCA:
        # Considering X = Y, MCA is the same as PCA. In this case,
        # singular_values_X is equivalent to the singular values obtained
        # when performing PCA of X
        self._singular_values = svalues
        self._explained_covariance = svalues
        self._squared_covariance_fraction = np.power(svalues, 2) / self._squared_total_variance
        self._singular_values_X = np.sqrt(svalues * (self.n_samples - 1))
        self._norm_X = np.sqrt(svalues)
        self._norm_Y = np.sqrt(svalues)

        # Ensure consistent signs for deterministic output
        maxidx = [abs(self._Vx).argmax(axis=0)]
        flip_signs = np.sign(self._Vx[maxidx, range(self._Vx.shape[1])])
        self._Vx *= flip_signs
        self._Vy *= flip_signs

        # Project data onto the singular vectors
        sqrt_expvar = np.sqrt(self._explained_covariance)
        self._Ux = self.X @ self._Vx / sqrt_expvar
        self._Uy = self.Y @ self._Vy / sqrt_expvar

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
            Vx = self._Vx * np.sqrt(self._explained_covariance)
            Vy = self._Vy * np.sqrt(self._explained_covariance)
        elif scaling == 2:
            Vx = self._Vx * self._explained_covariance
            Vy = self._Vy * self._explained_covariance
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
            Ux = self._Ux * np.sqrt(self._explained_covariance)
            Uy = self._Uy * np.sqrt(self._explained_covariance)
        elif scaling == 2:
            Ux = self._Ux * self._explained_covariance
            Uy = self._Uy * self._explained_covariance
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
        corr_X = self._corr_coef(self.X, self._Ux)
        corr_Y = self._corr_coef(self.Y, self._Uy)

        # Compute two-sided p-values
        # Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html#r8c6348c62346-1
        a = self.n_samples / 2 - 1
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
        corr_X = self._corr_coef(self.X, self._Uy)
        corr_Y = self._corr_coef(self.Y, self._Ux)

        # Compute two-sided p-values
        # Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html#r8c6348c62346-1
        a = self.n_samples / 2 - 1
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

        Xrec = Ux @ Vx.T
        Yrec = Uy @ Vy.T
        # Unweight and add mean
        Xrec = (Xrec / self._weights_X) + self._X_mean
        Yrec = (Yrec / self._weights_Y) + self._Y_mean
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
        # Remove mean and apply weights
        try:
            X -= self._X_mean
        except ValueError:
            err_msg = (
                'New data has invalid feature dimensions and cannot be '
                'projected onto singular vectors. Expected shapes'
            )
            raise ValueError(err_msg)
        X *= self._weights_X

        # Project new data onto singular values
        Ux = X @ self._Vx / np.sqrt(self._singular_values)

        # Apply scaling
        if scaling == 0:
            pass
        elif scaling == 1:
            Ux *= np.sqrt(self._singular_values)
        elif scaling == 2:
            Ux *= self._singular_values
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
        # Remove mean and apply weights
        try:
            Y -= self._Y_mean
        except ValueError:
            err_msg = (
                'New data has invalid feature dimensions and cannot be '
                'projected onto singular vectors. Expected shapes'
            )
            raise ValueError(err_msg)
        Y *= self._weights_Y

        # Project new data onto singular values
        Uy = Y @ self._Vy / np.sqrt(self._singular_values)

        # Apply scaling
        if scaling == 0:
            pass
        elif scaling == 1:
            Uy *= np.sqrt(self._singular_values)
        elif scaling == 2:
            Uy *= self._singular_values
        else:
            err_msg = (
                'Scaling option {:} is not valid but must be one '
                'of [0, 1, 2]'
            )
            err_msg = err_msg.foramt(scaling)
            raise ValueError(err_msg)
        return Uy
