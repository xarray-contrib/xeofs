from abc import ABC, abstractmethod


class _BaseStacker(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, da, sample_dims, feature_dims):
        'Define sample and feature dimensions'
        pass

    @abstractmethod
    def transform(self, da):
        pass

    @abstractmethod
    def inverse_transform_data(self, da):
        pass
    
    @abstractmethod
    def inverse_transform_components(self, da):
        pass
    
    @abstractmethod
    def inverse_transform_scores(self, da):
        pass