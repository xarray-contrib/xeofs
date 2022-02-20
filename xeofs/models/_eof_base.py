from abc import abstractmethod
from typing import Iterable, Optional

import numpy as np
from sklearn.decomposition import PCA


class _EOF_base():
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
        X -= X.mean(axis=0)

        # Weights are applied to features, not samples.
        if weights is None:
            # Use int type to ensure that there won't be rounding errors
            # when applying trivial weighting (= all weights equal 1)
            weights = np.ones(X.shape[1], dtype=int)

        # Standardization is included as weights
        if norm:
            stdev = X.std(axis=0)
            if (stdev == 0).any():
                err_msg = (
                    'Standard deviation of one ore more features is zero, '
                    'normalization not possible.'
                )
                raise ValueError(err_msg)
            weights = weights / stdev
        X = X * weights

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

    def eofs(self) -> np.ndarray:
        '''Get the EOFs.

        The empirical orthogonal functions (EOFs) are equivalent to the eigenvectors
        of the covariance matrix of `X`.

        '''

        return self._eofs

    def pcs(self) -> np.ndarray:
        '''Get the PCs.

        The principal components (PCs), also known as PC scores, are computed
        by projecting the data matrix `X` onto the eigenvectors.

        '''

        return self._pcs
