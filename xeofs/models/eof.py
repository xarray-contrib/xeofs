from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

from xeofs.models._eof_base import _EOF_base
from xeofs.models._array_transformer import _ArrayTransformer

import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


class EOF(_EOF_base):
    '''EOF analysis of a single ``np.ndarray``.

    Parameters
    ----------
    X : np.ndarray
        Data to be decpomposed. ``X`` can be any N-dimensional array with the
        first dimension containing the variable whose variance is to be
        maximised. All remaining dimensions will be automatically reshaped to
        obtain a 2D matrix. For example, given data of `n` time series of ``p1`` different
        variables each measured at ``p2`` different locations, the data matrix
        ``X`` of dimensions ``(n x p1 x p2)`` will maximise `temporal` variance.
        In contrast, if provided as ``(p2 x n x p1)``, the `spatial` variance
        will be maximised.
    n_modes : Optional[int]
        Number of modes to compute. Computing less modes can results in
        performance gains. If None, then the maximum number of modes is
        equivalent to ``min(n_samples, n_features)`` (the default is None).
    norm : bool
        Normalize each feature (e.g. grid cell) by its temporal standard
        deviation (the default is False).


    Examples
    --------

    Import package and create data:

    >>> import numpy as np
    >>> from xeofs.models import EOF
    >>> rng = np.random.default_rng(7)
    >>> X = rng.standard_normal((4, 2, 3))

    Initialize standardized EOF analysis and compute the first 2 modes:

    >>> model = EOF(X, norm=True, n_modes=2)
    >>> model.solve()

    Get explained variance:

    >>> model.explained_variance()
    ... array([4.81848833, 2.20765019])

    Get EOFs:

    >>> model.eofs()
    ... array([[[ 0.52002146, -0.00698575],
    ...         [ 0.41059796, -0.31603563]],
    ...
    ...         [[ 0.32091783,  0.47647144],
    ...         [-0.51771611,  0.01380736]],
    ...
    ...         [[-0.28840959,  0.63341346],
    ...         [ 0.32678537,  0.52119516]]])

    Get PCs:

    >>> model.pcs()
    ... array([[ 1.76084189, -0.92030205],
    ...        [-2.13581896, -1.49704775],
    ...        [ 2.02091078,  0.65502667],
    ...        [-1.64593371,  1.76232312]])

    '''

    def __init__(
        self,
        X: np.ndarray,
        n_modes : Optional[int] = None,
        norm : bool = False
    ):

        self._arr_tf = _ArrayTransformer()
        self.X = self._arr_tf.fit_transform(X)

        if norm:
            self.X /= self.X.std(axis=0)

        super().__init__(
            X=self.X,
            n_modes=n_modes,
            norm=norm
        )

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

        return self._arr_tf.back_transform(self._eofs.T).T

    def pcs(self):
        '''Get the PCs.

        The principal components (PCs), also known as PC scores, are computed
        by projecting the data matrix `X` onto the eigenvectors.

        '''

        return self._pcs
