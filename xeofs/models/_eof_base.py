from abc import abstractmethod
from typing import Iterable

import numpy as np


class _EOF_base():

    def __init__(
        self,
        X: Iterable[np.ndarray],
        Y: Iterable[np.ndarray] = None,
        n_modes=None,
        norm=False
    ):

        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        self.n_modes = n_modes
        if n_modes is None:
            self.n_modes = min(self.n_samples, self.n_features)

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
