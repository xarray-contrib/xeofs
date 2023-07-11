from abc import ABC, abstractmethod


class _BaseScaler(ABC):
    def __init__(self, with_std=True, with_coslat=False, with_weights=False):
        self._params = dict(
            with_std=with_std,
            with_coslat=with_coslat,
            with_weights=with_weights
        )

        self.mean = None
        self.std = None
        self.coslat_weights = None
        self.weights = None

    @abstractmethod
    def fit(self, X, sample_dims, feature_dims, weights=None):
        raise NotImplementedError
    
    @abstractmethod
    def transform(self, X):
        raise NotImplementedError
    
    @abstractmethod
    def fit_transform(self, X, sample_dims, feature_dims, weights=None):
        raise NotImplementedError
    
    @abstractmethod
    def inverse_transform(self, X):
        raise NotImplementedError
    
    def get_params(self):
        return self._params.copy()