from abc import ABC

from dask.diagnostics.progress import ProgressBar

from ..utils.data_types import DataArray


class _BaseModelDataContainer(ABC):
    '''Abstract base class that holds the model data.

    '''
    def __init__(self, input_data: DataArray, components: DataArray, scores: DataArray):
        
        self._verify_dims(input_data, ('sample', 'feature'))
        self._verify_dims(components, ('feature', 'mode'))
        self._verify_dims(scores, ('sample', 'mode'))
        
        components.name = 'components'
        scores.name = 'scores'

        self._input_data = input_data
        self._components = components
        self._scores = scores

    @staticmethod
    def _verify_dims(da: DataArray, dims: tuple):
        '''Verify that the dimensions of the data are correct.'''
        if not set(da.dims) == set(dims):
            raise ValueError(f'The data must have dimensions {dims}.')

    @property
    def input_data(self) -> DataArray:
        '''Get the input data.'''
        return self._input_data

    @property
    def components(self) -> DataArray:
        '''Get the components.'''
        return self._components
    
    @property
    def scores(self) -> DataArray:
        '''Get the scores.'''
        return self._scores
    
    def compute(self, verbose=False):
        '''Compute the results.'''
        if verbose:
            with ProgressBar():
                self._components = self._components.compute()
                self._scores = self._scores.compute()
        else:
            self._components = self._components.compute()
            self._scores = self._scores.compute()
    
    def set_attrs(self, attrs: dict):
        '''Set the attributes of the results.'''
        self._components.attrs.update(attrs)
        self._scores.attrs.update(attrs)