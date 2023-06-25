from abc import ABC, abstractmethod

import xarray as xr

from xeofs.models.scaler import Scaler, ListScaler
from xeofs.models.stacker import DataArrayStacker, DataArrayListStacker, DatasetStacker
from xeofs.models.decomposer import Decomposer
from ..utils.data_types import DataArray, DataArrayList, Dataset, XarrayData
from ..utils.tools import get_dims


class _BaseModel(ABC):

    def __init__(self, n_modes=10, standardize=False, use_coslat=False, use_weights=False, **kwargs):
        self._params = {
            'n_modes': n_modes,
            'standardize': standardize,
            'use_coslat': use_coslat,
            'use_weights': use_weights
        }
        self._scaling_params = {
            'with_std': standardize,
            'with_coslat': use_coslat,
            'with_weights': use_weights
        }

    @abstractmethod
    def fit(self, data, dims, weights=None):
        # Set sample and feature dimensions
        sample_dims, feature_dims = get_dims(data, sample_dims=dims)
        self.dims = {'sample': sample_dims, 'feature': feature_dims}
        
        # Scale the data
        self.scaler = self._create_scaler(data, **self._scaling_params)
        self.scaler.fit(data, sample_dims, feature_dims, weights)
        data = self.scaler.transform(data)

        # Stack the data
        self.stacker = self._create_stacker(data)
        self.stacker.fit(data, sample_dims, feature_dims)
        self.data = self.stacker.transform(data)
    
    @abstractmethod
    def singular_values(self):
        raise NotImplementedError()
    
    @abstractmethod
    def explained_variance(self):
        raise NotImplementedError()
    
    @abstractmethod
    def explained_variance_ratio(self):
        raise NotImplementedError()

    @abstractmethod
    def components(self):
        raise NotImplementedError()
    
    @abstractmethod
    def scores(self):
        raise NotImplementedError()
    
    def _create_scaler(self, data: XarrayData | DataArrayList, **kwargs):
        if isinstance(data, (xr.DataArray, xr.Dataset)):
            return Scaler(**kwargs)
        elif isinstance(data, list):
            return ListScaler(**kwargs)
        else:
            raise ValueError(f'Cannot scale data of type: {type(data)}')
    
    def _create_stacker(self, data: XarrayData | DataArrayList, **kwargs):
        if isinstance(data, xr.DataArray):
            return DataArrayStacker(**kwargs)
        elif isinstance(data, list):
            return DataArrayListStacker(**kwargs)
        elif isinstance(data, xr.Dataset):
            return DatasetStacker(**kwargs)
        else:
            raise ValueError(f'Cannot stack data of type: {type(data)}')

    def get_params(self):
        return self._params

class EOF(_BaseModel):

    def __init__(self, n_modes=10, standardize=False, use_coslat=False, **kwargs):
        super().__init__(n_modes, standardize, use_coslat, **kwargs)

    def fit(self, data: XarrayData | DataArrayList, dims, weights=None):
        
        super().fit(data, dims, weights)

        n_modes = self._params['n_modes']
        decomposer = Decomposer(n_components=n_modes)
        decomposer.fit(self.data)

        self._singular_values = decomposer.singular_values_
        self._components = decomposer.components_
        self._scores = decomposer.scores_

    def singular_values(self):
        return self._singular_values
    
    def explained_variance(self):
        raise NotImplementedError()
    
    def explained_variance_ratio(self):
        raise NotImplementedError()
    
    def components(self):
        return self.stacker.inverse_transform_components(self._components)
    
    def scores(self):
        return self.stacker.inverse_transform_scores(self._scores)