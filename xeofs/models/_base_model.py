import warnings
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import xarray as xr
import scipy as sc
from dask.diagnostics.progress import ProgressBar

from ..preprocessing.scaler import Scaler, ListScaler
from ..preprocessing.stacker import DataArrayStacker, DataArrayListStacker, DatasetStacker
from ..utils.data_types import DataArray, DataArrayList, Dataset, XarrayData
from ..utils.xarray_utils import get_dims
from .._version import __version__

# Ignore warnings from numpy casting with additional coordinates
warnings.filterwarnings("ignore", message=r"^invalid value encountered in cast*")


class _BaseModel(ABC):
    '''
    Abstract base class for EOF model. 

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
        self.attrs = {'model': 'BaseModel'}
        self.attrs.update(self._params)
        self.attrs.update({
            'software': 'xeofs',
            'version': __version__,
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

    def _preprocessing(self, data, dim, weights=None):
        '''Preprocess the data.
        
        This will scale and stack the data.
        
        Parameters:
        -------------
        data: xr.DataArray or list of xarray.DataArray
            Input data.
        dim: tuple
            Tuple specifying the sample dimensions. The remaining dimensions
            will be treated as feature dimensions.
        weights: xr.DataArray or xr.Dataset or None, default=None
            If specified, the input data will be weighted by this array.
        
        '''
        # Set sample and feature dimensions
        sample_dims, feature_dims = get_dims(data, sample_dims=dim)
        self.dims = {'sample': sample_dims, 'feature': feature_dims}
        
        # Scale the data
        self.scaler = self._create_scaler(data, **self._scaling_params)
        self.scaler.fit(data, sample_dims, feature_dims, weights)  # type: ignore
        data = self.scaler.transform(data)

        # Stack the data
        self.stacker = self._create_stacker(data)
        self.data = self.stacker.fit_transform(data, sample_dims, feature_dims)  # type: ignore

    @abstractmethod
    def fit(self, data, dim, weights=None):
        '''
        Abstract method to fit the model.

        Parameters:
        -------------
        data: xr.DataArray or list of xarray.DataArray
            Input data.
        dim: tuple
            Tuple specifying the sample dimensions. The remaining dimensions 
            will be treated as feature dimensions.
        weights: xr.DataArray or xr.Dataset or None, default=None
            If specified, the input data will be weighted by this array.

        '''
        # Here follows the implementation to fit the model
        # Typically you want to start by calling self._preprocessing(data, dim, weights)
        # ATTRIBUTES TO BE DEFINED:
        self._total_variance = None
        self._singular_values = None
        self._explained_variance = None
        self._explained_variance_ratio = None
        self._components = None
        self._scores = None

    @abstractmethod
    def transform(self):
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self):
        raise NotImplementedError

    def singular_values(self):
        '''Return the singular values of the model.

        Returns:
        ----------
        singular_values: DataArray
            Singular values of the fitted model.

        '''
        return self._singular_values
    
    def explained_variance(self):
        '''Return explained variance.'''
        return self._explained_variance
    
    def explained_variance_ratio(self):
        '''Return explained variance ratio.'''
        return self._explained_variance_ratio

    def components(self):
        '''Return the components.
        
        The components in EOF anaylsis are the eigenvectors of the covariance matrix
        (or correlation) matrix. Other names include the principal components or EOFs.

        Returns:
        ----------
        components: DataArray | Dataset | List[DataArray]
            Components of the fitted model.

        '''
        return self.stacker.inverse_transform_components(self._components)  #type: ignore
    
    def scores(self):
        '''Return the scores.
        
        The scores in EOF anaylsis are the projection of the data matrix onto the 
        eigenvectors of the covariance matrix (or correlation) matrix. 
        Other names include the principal component (PC) scores or just PCs.

        Returns:
        ----------
        components: DataArray | Dataset | List[DataArray]
            Scores of the fitted model.

        '''
        return self.stacker.inverse_transform_scores(self._scores)  #type: ignore

    def get_params(self):
        return self._params

    def compute(self, verbose: bool = False):
        '''Computing the model will load and compute Dask arrays.
        
        Parameters:
        -------------
        verbose: bool, default=False
            If True, print information about the computation process.
            
        '''

        if verbose:
            with ProgressBar():
                print('Computing STANDARD MODEL...')
                print('-'*80)
                print('Total variance...')
                self._total_variance = self._total_variance.compute()  # type: ignore
                print('Singular values...')
                self._singular_values = self._singular_values.compute()   # type: ignore
                print('Explained variance...')
                self._explained_variance = self._explained_variance.compute()   # type: ignore
                print('Explained variance ratio...')
                self._explained_variance_ratio = self._explained_variance_ratio.compute()   # type: ignore
                print('Components...')
                self._components = self._components.compute()    # type: ignore
                print('Scores...')
                self._scores = self._scores.compute()    # type: ignore
        else:
            self._total_variance = self._total_variance.compute()  # type: ignore
            self._singular_values = self._singular_values.compute()   # type: ignore
            self._explained_variance = self._explained_variance.compute()   # type: ignore
            self._explained_variance_ratio = self._explained_variance_ratio.compute()   # type: ignore
            self._components = self._components.compute()    # type: ignore
            self._scores = self._scores.compute()    # type: ignore

    def _assign_meta_data(self):
        '''Set attributes to the model output.'''
        self._total_variance.attrs.update(self.attrs)  # type: ignore
        self._singular_values.attrs.update(self.attrs)  # type: ignore
        self._explained_variance.attrs.update(self.attrs)  # type: ignore
        self._explained_variance_ratio.attrs.update(self.attrs)  # type: ignore
        self._components.attrs.update(self.attrs)  # type: ignore
        self._scores.attrs.update(self.attrs)  # type: ignore
