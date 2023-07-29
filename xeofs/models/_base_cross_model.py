from abc import ABC, abstractmethod
from datetime import datetime

from dask.diagnostics.progress import ProgressBar

from ..preprocessing.preprocessor import Preprocessor
from ..data_container import _BaseCrossModelDataContainer
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
    def __init__(self, n_modes=10, standardize=False, use_coslat=False, use_weights=False):
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
            'version': __version__,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        # Initialize preprocessors to scale and stack left (1) and right (2) data
        self.preprocessor1 = Preprocessor(
            with_std=standardize,
            with_coslat=use_coslat,
            with_weights=use_weights,
        )
        self.preprocessor2 = Preprocessor(
            with_std=standardize,
            with_coslat=use_coslat,
            with_weights=use_weights,
        )
        # Initialize the data container only to avoid type errors
        # The actual data container will be initialized in respective subclasses
        self.data: _BaseCrossModelDataContainer = _BaseCrossModelDataContainer()

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
        # Typically you want to start by calling
        # self.preprocessor1.fit_transform(data1, dim, weights)
        # self.preprocessor2.fit_transform(data2, dim, weights)
        raise NotImplementedError
    
    @abstractmethod
    def transform(self, data1, data2):
        raise NotImplementedError
    
    @abstractmethod
    def inverse_transform(self, mode):
        raise NotImplementedError
    
    def components(self):
        '''Get the components.'''
        return self.data.components1, self.data.components2
    
    def scores(self):
        '''Get the scores.'''
        return self.data.scores1, self.data.scores2

    def compute(self, verbose=False):
        '''Compute the results.'''
        if verbose:
            with ProgressBar():
                self.data.compute()
        else:
            self.data.compute()

    def get_params(self):
        return self._params
    