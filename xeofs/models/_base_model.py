import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from dask.diagnostics.progress import ProgressBar

from ..preprocessing.preprocessor import Preprocessor
from ..data_container import _BaseModelDataContainer
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
    def __init__(self, n_modes=10, standardize=False, use_coslat=False, use_weights=False):
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
        # Initialize the data container only to avoid type errors
        # The actual data container will be initialized in respective subclasses
        self.data: _BaseModelDataContainer = _BaseModelDataContainer()

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
        raise NotImplementedError

    @abstractmethod
    def transform(self):
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self):
        raise NotImplementedError

    def components(self):
        '''Get the components.'''
        return self.data.components
    
    def scores(self):
        '''Get the scores.'''
        return self.data.scores

    def compute(self, verbose=False):
        '''Compute and load delayed model results.
        
        Parameters
        ----------
        verbose : bool
            Whether or not to provide additional information about the computing progress.
            
        '''
        if verbose:
            with ProgressBar():
                self.data.compute()
        else:
            self.data.compute()

    def get_params(self):
        return self._params
    