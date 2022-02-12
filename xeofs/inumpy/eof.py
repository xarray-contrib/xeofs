from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA

from xeofs.inumpy._eof_base import _EOF_base


class EOF(_EOF_base):

    def __init__(
        self,
        X: Iterable[np.ndarray],
        Y: Iterable[np.ndarray] = None,
        n_modes=None,
        norm=False,
        axis=0
    ):
        super().__init__(
            X=X,
            Y=None,
            n_modes=n_modes,
            norm=norm,
            axis=axis
        )

    def solve(self):
        pca = PCA(n_components=self._n_modes)
        self._pcs = pca.fit_transform(self.X)
        self._singular_values = pca.singular_values_
        self._explained_variance = pca.explained_variance_
        self._explained_variance_ratio_  = pca.explained_variance_ratio_
        self._eofs = pca.components_.T

    def singular_values(self):
        return self._singular_values

    def explained_variance(self):
        return self._explained_variance

    def explained_variance_ratio(self):
        return self._explained_variance_ratio

    def eofs(self):
        out_shape = np.append(self._feature_space, -1)
        return self._eofs.reshape(out_shape)

    def pcs(self):
        return self._pcs
