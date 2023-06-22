from abc import ABC, abstractmethod


class _BaseScaler(ABC):
    def __init__(self, sample_dims, feature_dims, with_copy=True, with_mean=True, with_std=True, with_coslat=False, weights=None):
        self.mean = None
        self.std = None
        self.coslat_weights = None
        self.weights = weights

        self._params = dict(
            dims=dict(sample=sample_dims, feature=feature_dims),
            with_copy=with_copy,
            with_mean=with_mean,
            with_std=with_std,
            with_coslat=with_coslat,
            with_weights=True if weights is not None else False
        )

    @abstractmethod
    def fit(self, X):
        raise NotImplementedError
    
    @abstractmethod
    def transform(self, X):
        raise NotImplementedError
    
    @abstractmethod
    def inverse_transform(self, X):
        raise NotImplementedError