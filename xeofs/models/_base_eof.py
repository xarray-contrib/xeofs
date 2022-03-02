from typing import Optional, Tuple, List, Union

import numpy as np
import scipy as sc
from sklearn.decomposition import PCA

from ..utils.tools import get_mode_selector


class _BaseEOF():
    '''Base class for univariate EOF analysis.

    Parameters
    ----------
    X : np.ndarray
        2D data matrix ``X`` with samples as rows and features as columns.
        NaN entries will raise an error.
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
        n_modes: Optional[int] = None,
        norm: bool = False,
        weights: Optional[np.ndarray] = None
    ):
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

        self.n_modes = n_modes
        if n_modes is None:
            self.n_modes = min(self.n_samples, self.n_features)

        self.X = X

    def solve(self) -> None:
        '''
        Perform the EOF analysis.

        To boost performance, the standard solver is based on
        the PCA implementation of scikit-learn [1]_ which uses different algorithms
        to perform the decomposition based on the data matrix size.

        Naive approaches using singular value decomposition of the
        data matrix ``X (n x p)`` or the covariance matrix ``C (p x p)``
        quickly become infeasable computationally when the number of
        samples :math:`n` or features :math:`p` increase (computational power increases
        by :math:`O(n^2p)` and :math:`O(p^3)`, respectively.)


        References
        ----------
        .. [1] https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

        '''

        pca = PCA(n_components=self.n_modes)
        self._pcs = pca.fit_transform(self.X)
        self._singular_values = pca.singular_values_
        self._explained_variance = pca.explained_variance_
        self._explained_variance_ratio  = pca.explained_variance_ratio_
        self._eofs = pca.components_.T

        # Normalize PCs so they are orthonormal
        # Note: singular values = sqrt(explained_variance * (n_samples - 1))
        self._pcs = self._pcs / self._singular_values

        # Ensure consistent signs for deterministic output
        maxidx = [abs(self._eofs).argmax(axis=0)]
        flip_signs = np.sign(self._eofs[maxidx, range(self._eofs.shape[1])])
        self._eofs *= flip_signs
        self._pcs *= flip_signs

    def singular_values(self) -> np.ndarray:
        '''Get the singular values.

        The `i` th singular value :math:`\sigma_i` is defined by

        .. math::
           \sigma_i = \sqrt{n \lambda_i}

        where :math:`\lambda_i` and :math:`n` are the associated eigenvalues
        and the number of samples, respectively.

        '''
        return self._singular_values

    def explained_variance(self) -> np.ndarray:
        '''Get the explained variance.

        The explained variance is simply given by the individual eigenvalues
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

    def eofs(self, scaling : int = 0) -> np.ndarray:
        '''Get the EOFs.

        The empirical orthogonal functions (EOFs) are equivalent to the
        eigenvectors of the covariance matrix of `X`.

        Parameters
        ----------
        scaling : [0, 1, 2]
            EOFs are scaled (i) to be orthonormal (``scaling=0``), (ii) by the
            square root of the eigenvalues (``scaling=1``) or (iii) by the
            singular values (``scaling=2``). In case no weights were applied,
            scaling by the singular values results in the EOFs having the
            unit of the input data (the default is 0).

        '''
        if scaling == 0:
            eofs = self._eofs
        elif scaling == 1:
            eofs = self._eofs * np.sqrt(self._explained_variance)
        elif scaling == 2:
            eofs = self._eofs * self._singular_values
        return eofs

    def pcs(self, scaling : int = 0) -> np.ndarray:
        '''Get the PCs.

        The principal components (PCs), also known as PC scores, are computed
        by projecting the data matrix `X` onto the eigenvectors.

        Parameters
        ----------
        scaling : [0, 1, 2]
            PCs are scaled (i) to be orthonormal (``scaling=0``), (ii) by the
            square root of the eigenvalues (``scaling=1``) or (iii) by the
            singular values (``scaling=2``). In case no weights were applied,
            scaling by the singular values results in the PCs having the
            unit of the input data (the default is 0).

        '''
        if scaling == 0:
            pcs = self._pcs
        elif scaling == 1:
            pcs = self._pcs * np.sqrt(self._explained_variance)
        elif scaling == 2:
            pcs = self._pcs * self._singular_values
        return pcs

    def eofs_as_correlation(self) -> Tuple[np.ndarray, np.ndarray]:
        '''Correlation coefficients between PCs and data matrix.

        Returns
        -------
        Tuple[np.ndarray, np.ndarry]
            Matrices of correlation coefficients and associated
            two-sided p-values with features as rows and modes as columns.

        '''

        # Compute correlation matrix
        corr = np.corrcoef(self.X, self._pcs, rowvar=False)
        corr = corr[:self.n_features, self.n_features:]
        # Compute two-sided p-values
        # Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html#r8c6348c62346-1
        a = self.n_samples / 2 - 1
        dist = sc.stats.beta(a, a, loc=-1, scale=2)
        pvals = 2 * dist.cdf(-abs(corr))
        return corr, pvals

    def reconstruct_X(
        self,
        mode : Optional[Union[int, List[int], slice]] = None
    ) -> np.ndarray:
        '''Reconstruct original data field ``X`` using the PCs and EOFs.

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
        eofs = self._eofs
        pcs = self._pcs * self._singular_values
        # Select modes to reconstruct X
        mode = get_mode_selector(mode)
        eofs = eofs[:, mode]
        pcs = pcs[:, mode]
        Xrec = pcs @ eofs.T
        # Unweight and add mean
        return (Xrec / self._weights) + self._X_mean

    def project_onto_eofs(
        self,
        X : np.ndarray, scaling : int = 0
    ) -> np.ndarray:
        '''Project new data onto the EOFs.

        Parameters
        ----------
        X : np.ndarray
             New data to project. Data must be a 2D matrix.
        scaling : [0, 1, 2]
            Projections are scaled (i) to be orthonormal (``scaling=0``), (ii) by the
            square root of the eigenvalues (``scaling=1``) or (iii) by the
            singular values (``scaling=2``). In case no weights were applied,
            scaling by the singular values results in the projections having the
            unit of the input data (the default is 0).

        '''
        dof = self.n_samples - 1
        try:
            X -= self._X_mean
        except ValueError:
            err_msg = (
                'New data has invalid feature dimensions and cannot be '
                'projected onto EOFs.'
            )
            raise ValueError(err_msg)
        X *= self._weights
        pcs = X @ self._eofs / np.sqrt(self._explained_variance * dof)
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
