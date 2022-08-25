from typing import Optional, Tuple, List, Union

import numpy as np
from scipy.signal import hilbert

from ..utils.rotation import promax
from ..utils.distance_matrix import distance_matrix


class _BaseROCK_PCA():
    '''Base class for ROCK-PCA.'''

    def __init__(
        self,
        X: np.ndarray,
        n_rot : int,
        power : int,
        sigma : float,
        n_modes: Optional[int] = None,
        norm: bool = False,
        weights: Optional[np.ndarray] = None,
    ):
        '''
        Perform rotated complex kernel PCA (ROCK-PCA).

        ROCK-PCA is a non-linear complex decomposition with an additional
        rotation in the Fourier space.

        Original reference:
        - Bueso, D., Piles, M. & Camps-Valls, G. Nonlinear PCA for Spatio-Temporal
        Analysis of Earth Observation Data. IEEE Trans. Geosci. Remote Sensing 1–12
        (2020) doi:10.1109/TGRS.2020.2969813.

        Parameters
        ----------
        X : np.ndarray
            2D data matrix ``X`` with samples as rows and features as columns.
            NaN entries will raise an error.
        n_rot : int
            Number of PCs to be rotated.
        power : int
            Power parameter for Promax rotation. If ``power=0``, no rotation is performed.
            For ``power=1``, this is equivalent to Varimax rotation.
        sigma : float,
            Hyperparamter defining the "width" Gaussian kernel
        n_modes : Optional[int]
            Number of modes to be retrieved. If None, then all possible modes will
            be computed. Reducing ``n_modes`` can greatly speed up computational
            (the default is None).
        norm : bool
            Normalize each feature by its standard deviation (the default is False).
        weights : Optional[np.ndarray]
            Apply `weights` to features. Weights must be a 1D array with the same
            length as number of features. No NaN entries are allowed
            (the default is None).

        References
        ----------
        .. [1] https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8989964&casa_token=3zKG0dtp-ewAAAAA:FM1CrVISSSqhWEAwPGpQqCgDYccfLG4N-67xNNDzUBQmMvtIOHuC7T6X-TVQgbDg3aDOpKBksg&tag=1
        .. [2] https://github.com/DiegoBueso/ROCK-PCA

        '''
        self._params = {
            'n_rot': n_rot,
            'power': power,
            'sigma': sigma,
            'norm': norm
        }
        # Remove mean for each feature
        self._X_mean = X.mean(axis=0)
        X -= self._X_mean

        # Weights are applied to features, not samples.
        self._weights = weights
        if self._weights is None:
            # Use int type to ensure that there won't be rounding errors
            # when applying trivial weighting (= all weights equal 1)
            self._weights = np.ones(X.shape[1], dtype=int)

        # Standardization is included as weights
        if norm:
            stdev = X.std(axis=0)
            if (stdev == 0).any():
                err_msg = (
                    'Standard deviation of one ore more features is zero, '
                    'normalization not possible.'
                )
                raise ValueError(err_msg)
            self._weights = self._weights / stdev
        X = X * self._weights

        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        if n_modes is None:
            n_modes = min(self.n_samples, self.n_features)
        self._params['n_modes'] = min(n_modes, n_rot)

        # Complexify data using Hilbert transform
        self.X = hilbert(X, axis=0)
        self.X -= self.X.mean(axis=0)

    def solve(self, algorithm='original') -> None:
        '''
        Perform complex non-linear decomposition.

        '''
        # Sigma estimation from median distance
        print('Build distance matrix ... ', flush=True)
        sigma = self._params['sigma']
        n_rot = self._params['n_rot']
        power = self._params['power']
        n_modes = self._params['n_modes']

        distance = distance_matrix(self.X.T, self.X.T, algorithm)
        print('Build kernel matrix ... ', flush=True)
        kernel = np.exp(-distance / (2 * (sigma**2)))
        center_matrix = np.identity(self.n_samples) - (1. / self.n_samples) * np.ones(self.n_samples)
        kernel = center_matrix @ kernel @ center_matrix
        print('SVD ... ', flush=True)
        U, lambdas, _ = np.linalg.svd(kernel, full_matrices=False)
        self.lambdas = lambdas

        # Rotation in frequency space
        U = U[:, :n_rot]
        if power > 0:
            print('Rotation ... ', flush=True)
            Ufft = np.fft.fft(U, axis=0)
            # Varimax
            U, _, _    = promax(Ufft, power=1)
            # Promax according to Bueso et al. implementation:
            # https://github.com/DiegoBueso/ROCK-PCA/blob/master/rock-code/Promax.m
            U = np.sign(U) * abs(U)**power
            # Renormalization
            U = abs(U) / abs(Ufft) * Ufft
            U = np.real(np.fft.ifft(U, axis=0))
            U = hilbert(U, axis=0)
            U = center_matrix @ U

        # Estimate variance explained
        self._explained_variance = np.linalg.norm(U.conj().T @ self.X, axis=1) / np.linalg.norm(U, axis=0)
        self._total_variance = self._explained_variance.sum()
        self._explained_variance_ratio = self._explained_variance / self._total_variance

        # Reorder such that first mode explained most variance
        idx_sort = np.argsort(self._explained_variance)[::-1]
        self._explained_variance = self._explained_variance[idx_sort]
        self._explained_variance_ratio = self._explained_variance_ratio[idx_sort]
        self._pcs = U[:, idx_sort]

        self._explained_variance = self._explained_variance[:n_modes]
        self._explained_variance_ratio = self._explained_variance_ratio[:n_modes]
        self._pcs = self._pcs[:, :n_modes]

        # Spatial projections
        self._eofs = (self._pcs.conjugate().T @ self.X).T

        # Ensure consistent signs for deterministic output
        maxidx = [abs(self._eofs).argmax(axis=0)]
        flip_signs = np.sign(self._eofs[maxidx, range(self._eofs.shape[1])])
        self._eofs *= flip_signs
        self._pcs *= flip_signs

    def explained_variance(self) -> np.ndarray:
        '''Get the explained variance.

        The explained variance is given by the individual eigenvalues
        of the covariance matrix.

        '''
        return self._explained_variance

    def explained_variance_ratio(self) -> np.ndarray:
        '''Get the explained variance ratio.

        The explained variance ratio is the fraction of total variance
        explained by a given mode and is calculated by :math:`\lambda_i / \sum_i^m \lambda_i`
        where `m` is the total number of modes.

        '''

        return self._explained_variance_ratio

    def eofs(self) -> np.ndarray:
        '''Get the EOFs.

        '''
        return self._eofs

    def pcs(self) -> np.ndarray:
        '''Get the PCs.

        The principal components (PCs), also known as PC scores, are computed
        by projecting the data matrix `X` onto the EOFs.

        '''
        return self._pcs

    def eofs_amplitude(self):
        '''Get the EOF amplitude functions.

        The amplitude is defined as

        .. math::
            A = (V \\oplus \\bar{V})^{1/2}

        where :math:`V` denotes the complex EOFs, :math:`\\bar{V}` the complex
        conjugate and :math:`\\oplus` denotes the Hadamard (element-wise)
        product.

        '''
        return np.sqrt(self._eofs.conj() * self._eofs).real

    def pcs_amplitude(self):
        '''Get the PC amplitude functions.

        The amplitude is defined as

        .. math::
            A = (U \\oplus \\bar{U})^{1/2}

        where :math:`U` denotes the complex PCs, :math:`\\bar{U}` the complex
        conjugate and :math:`\\oplus` denotes the Hadamard (element-wise)
        product.

        '''
        return np.sqrt(self._pcs.conj() * self._pcs).real

    def eofs_phase(self):
        '''Get the EOF phase functions.

        The phase is defined as

        .. math::
            P = \\arctan(I(V) \\oslash R(V))

        where :math:`V` denotes the complex EOFs and :math:`R(\\cdot)`,
        :math:`I(\\cdot)` the real and imaginary part, respectively.
        The :math:`\\oslash` operator represents element-wise division.

        '''
        return np.arctan2(self._eofs.imag, self._eofs.real)

    def pcs_phase(self):
        '''Get the PC phase functions.

        The phase is defined as

        .. math::
            P = \\arctan(I(U) \\oslash R(U))

        where :math:`U` denotes the complex PCs and :math:`R(\\cdot)`,
        :math:`I(\\cdot)` the real and imaginary part, respectively.
        The :math:`\\oslash` operator represents element-wise division.

        '''
        return np.arctan2(self._pcs.imag, self._pcs.real)
