from abc import abstractmethod
from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA


class _EOF_base():

    def __init__(
        self,
        X: Iterable[np.ndarray],
        n_modes=None,
        norm=False
    ):

        if norm:
            X /= X.std(axis=0)

        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        self.n_modes = n_modes
        if n_modes is None:
            self.n_modes = min(self.n_samples, self.n_features)

        # TODO: weights

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

        # Consistent signs for deterministic output
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

    def explained_variance(self):
        '''Get the explained variance.

        The explained variance is simply given by the individual eigenvalues
        of the covariance matrix.

        '''

        return self._explained_variance

    def explained_variance_ratio(self):
        '''Get the explained variance ratio.

        The explained variance ratio is the fraction of total variance
        explained by a given mode and is calculated by :math:`\lambda_i / \sum_i^m \lambda_i`
        where `m` is the total number of modes.

        '''

        return self._explained_variance_ratio

    def eofs(self):
        '''Get the EOFs.

        The empirical orthogonal functions (EOFs) are equivalent to the eigenvectors
        of the covariance matrix of `X`.

        '''

        return self._eofs

    def pcs(self):
        '''Get the PCs.

        The principal components (PCs), also known as PC scores, are computed
        by projecting the data matrix `X` onto the eigenvectors.

        '''

        return self._pcs
