from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import xarray as xr
import dask.array as da

from ..preprocessing.scaler import Scaler, ListScaler
from ..preprocessing.stacker import DataArrayStacker, DataArrayListStacker, DatasetStacker
from ..utils.xarray_utils import get_dims
from ..utils.data_types import XarrayData, DataArrayList
from .._version import __version__

class _BaseCrossModel(ABC):
    '''
    Abstract base class for cross-decomposition models. 

    Parameters:
    -------------
    n_modes: int, default=10
        Number of modes to calculate.
    standardize: bool, default=False
        Whether to standardize the input data.
    use_coslat: bool, default=False
        Whether to use cosine of latitude for scaling.
    use_weights: bool, default=False
        Whether to use weights.

    '''
    def __init__(self, n_modes=10, standardize=False, use_coslat=False, use_weights=False, **kwargs):
        # Define model parameters
        self._params = {
            'n_modes': n_modes,
            'standardize': standardize,
            'use_coslat': use_coslat,
            'use_weights': use_weights
        }

        # Define analysis-relevant meta data
        self.attrs = {'model': 'BaseCrossModel'}
        self.attrs.update(self._params)
        self.attrs.update({
            'software': 'xeofs',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        # Some more parameters used for scaling
        self._scaling_params = {
            'with_std': standardize,
            'with_coslat': use_coslat,
            'with_weights': use_weights
        }
    
    @staticmethod
    def _create_scaler(data: XarrayData | DataArrayList, **kwargs):
        if isinstance(data, (xr.DataArray, xr.Dataset)):
            return Scaler(**kwargs)
        elif isinstance(data, list):
            return ListScaler(**kwargs)
        else:
            raise ValueError(f'Cannot scale data of type: {type(data)}')
    
    @staticmethod
    def _create_stacker(data: XarrayData | DataArrayList, **kwargs):
        if isinstance(data, xr.DataArray):
            return DataArrayStacker(**kwargs)
        elif isinstance(data, list):
            return DataArrayListStacker(**kwargs)
        elif isinstance(data, xr.Dataset):
            return DatasetStacker(**kwargs)
        else:
            raise ValueError(f'Cannot stack data of type: {type(data)}')

    def _preprocessing(self, data1, data2, dim, weights1=None, weights2=None):
        '''Preprocess the data.
        
        This will scale and stack the data.
        
        Parameters:
        -------------
        data1: xr.DataArray or list of xarray.DataArray
            Left input data.
        data2: xr.DataArray or list of xarray.DataArray
            Right input data.
        dim: tuple
            Tuple specifying the sample dimensions. The remaining dimensions
            will be treated as feature dimensions.
        weights1: xr.DataArray or xr.Dataset or None, default=None
            If specified, the left input data will be weighted by this array.
        weights2: xr.DataArray or xr.Dataset or None, default=None
            If specified, the right input data will be weighted by this array.
        
        '''
        # Set sample and feature dimensions
        sample_dims, feature_dims1 = get_dims(data1, sample_dims=dim)
        sample_dims, feature_dims2 = get_dims(data2, sample_dims=dim)
        self.dim = {'sample': sample_dims, 'feature1': feature_dims1, 'feature2': feature_dims2}
        
        # Scale the data
        self.scaler1 = self._create_scaler(data1, **self._scaling_params)
        self.scaler1.fit(data1, sample_dims, feature_dims1, weights1)  # type: ignore
        data1 = self.scaler1.transform(data1)

        self.scaler2 = self._create_scaler(data2, **self._scaling_params)
        self.scaler2.fit(data2, sample_dims, feature_dims2, weights2)  # type: ignore
        data2 = self.scaler2.transform(data2)

        # Stack the data
        self.stacker1 = self._create_stacker(data1)
        self.data1 = self.stacker1.fit_transform(data1, sample_dims, feature_dims1)  # type: ignore

        self.stacker2 = self._create_stacker(data2)
        self.data2 = self.stacker2.fit_transform(data2, sample_dims, feature_dims2)  # type: ignore

    @abstractmethod
    def fit(self, data1, data2, dim, weights1=None, weights2=None):
        '''
        Abstract method to fit the model.

        Parameters:
        -------------
        data1: xr.DataArray or list of xarray.DataArray
            Left input data.
        data2: xr.DataArray or list of xarray.DataArray
            Right input data.
        dim: tuple
            Tuple specifying the sample dimensions. The remaining dimensions 
            will be treated as feature dimensions.
        weights1: xr.DataArray or xr.Dataset or None, default=None
            If specified, the left input data will be weighted by this array.
        weights2: xr.DataArray or xr.Dataset or None, default=None
            If specified, the right input data will be weighted by this array.

        '''
        # Here follows the implementation to fit the model
        # Typically you want to start by calling self._preprocessing(data1, data2, dim, weights)
        # ATTRIBUTES TO BE DEFINED:

        raise NotImplementedError
    
    @abstractmethod
    def transform(self, data1, data2):
        raise NotImplementedError
    
    @abstractmethod
    def inverse_transform(self, mode):
        raise NotImplementedError

    def get_params(self):
        return self._params
    
    @abstractmethod
    def compute(self):
        '''Computing the model will load and compute Dask arrays.'''
        raise NotImplementedError