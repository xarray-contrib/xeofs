from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA

from xeofs.models._eof_base import _EOF_base
from xeofs.models._array_transformer import _ArrayTransformer

import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


class EOF(_EOF_base):

    def __init__(
        self,
        X: Iterable[np.ndarray],
        Y: Iterable[np.ndarray] = None,
        n_modes=None,
        norm=False
    ):

        self._arr_tf = _ArrayTransformer()
        self.X = self._arr_tf.fit_transform(X)

        if norm:
            self.X /= self.X.std(axis=0)

        super().__init__(
            X=self.X,
            Y=None,
            n_modes=n_modes,
            norm=norm
        )

    def solve(self):
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

    def singular_values(self):
        return self._singular_values

    def explained_variance(self):
        return self._explained_variance

    def explained_variance_ratio(self):
        return self._explained_variance_ratio

    def eofs(self):
        return self._arr_tf.back_transform(self._eofs.T).T

    def pcs(self):
        return self._pcs
