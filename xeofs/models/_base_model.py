import warnings
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import xarray as xr
import scipy as sc
from dask.diagnostics.progress import ProgressBar

from ..preprocessing.scaler_factory import ScalerFactory
from ..preprocessing.stacker_factory import StackerFactory
from ..preprocessing.stacker import SingleDataArrayStacker, ListDataArrayStacker, SingleDatasetStacker
from ..preprocessing.preprocessor import Preprocessor
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

        # Initialize the Preprocessor to scale and stack the data
        self.preprocessor = Preprocessor(
            with_std=standardize,
            with_coslat=use_coslat, 
            with_weights=use_weights
        )

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
        # Typically you want to start by calling the Preprocessor first:
        # self.preprocessor.fit_transform(data, dim, weights)


    @abstractmethod
    def transform(self):
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self):
        raise NotImplementedError

    def components(self):
        '''Return the components.
        
        The components in EOF anaylsis are the eigenvectors of the covariance matrix
        (or correlation) matrix. Other names include the principal components or EOFs.

        Returns:
        ----------
        components: DataArray | Dataset | List[DataArray]
            Components of the fitted model.

        '''
        components = self.data.components
        return self.preprocessor.inverse_transform_components(components)  #type: ignore
    
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
        scores = self.data.scores
        return self.preprocessor.inverse_transform_scores(scores)  #type: ignore

    def get_params(self):
        return self._params
    
    def compute(self, verbose=False):
        '''Compute the results.'''
        if verbose:
            with ProgressBar():
                self.data.compute() #type: ignore
        else:
            self.data.compute() #type: ignore