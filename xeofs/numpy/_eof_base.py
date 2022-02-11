from abc import abstractmethod
from typing import Iterable

import numpy as np


class _EOF_base():

    def __init__(
        self,
        X: Iterable[np.ndarray],
        Y: Iterable[np.ndarray] = None,
        n_modes=None,
        norm=False,
        axis=0
    ):
        self._X_shape = np.array(X.shape)
        self._feature_space = np.delete(self._X_shape, axis)

        n_samples = np.product(self._X_shape[axis])
        if len(self._X_shape) > 2:
            X = X.reshape(n_samples, -1)

        self.X = X
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        self._n_modes = n_modes
        if n_modes is None:
            self._n_modes = min(self.n_samples, self.n_features)

        # TODO: remove NaN
        # self._nan_idx = np.nan(X).any(axis=0)

        # TODO: normalization

        # TODO: weights

    @abstractmethod
    def solve(self):
        raise NotImplementedError

    @abstractmethod
    def singular_values(self):
        raise NotImplementedError

    @abstractmethod
    def explained_variance(self):
        raise NotImplementedError

    @abstractmethod
    def explained_variance_ratio(self):
        raise NotImplementedError

    @abstractmethod
    def eofs(self):
        raise NotImplementedError

    @abstractmethod
    def pcs(self):
        raise NotImplementedError
