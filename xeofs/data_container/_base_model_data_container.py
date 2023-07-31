from abc import ABC
from typing import Optional

from dask.diagnostics.progress import ProgressBar

from ..utils.data_types import DataArray


class _BaseModelDataContainer(ABC):
    '''Abstract base class that holds the model data.

    '''
    def __init__(self):
        self._input_data: Optional[DataArray] = None
        self._components: Optional[DataArray] = None
        self._scores: Optional[DataArray] = None

    @staticmethod
    def _verify_dims(da: DataArray, dims_expected: tuple):
        '''Verify that the dimensions of the data are correct.'''
        if not set(da.dims) == set(dims_expected):
            raise ValueError(f'The data must have dimensions {dims_expected}.')

    @staticmethod
    def _sanity_check(data) -> DataArray:
        '''Check whether the Data of the DataContainer has been set.'''
        if data is None:
            raise ValueError('There is no data. Have you called .fit()?')
        else:
            return data

    def set_data(self, input_data: DataArray, components: DataArray, scores: DataArray):
        
        self._verify_dims(input_data, ('sample', 'feature'))
        self._verify_dims(components, ('feature', 'mode'))
        self._verify_dims(scores, ('sample', 'mode'))
        
        components.name = 'components'
        scores.name = 'scores'

        self._input_data = input_data
        self._components = components
        self._scores = scores

    @property
    def input_data(self) -> DataArray:
        '''Get the input data.'''
        data = self._sanity_check(self._input_data)
        return data

    @property
    def components(self) -> DataArray:
        '''Get the components.'''
        components = self._sanity_check(self._components)
        return components
    
    @property
    def scores(self) -> DataArray:
        '''Get the scores.'''
        scores = self._sanity_check(self._scores)
        return scores
    
    def compute(self, verbose=False):
        '''Compute and load delayed dask DataArrays into memory.
        
        Parameters
        ----------
        verbose : bool
            Whether or not to provide additional information about the computing progress.
        '''
        if verbose:
            with ProgressBar():
                self._components = self.components.compute()
                self._scores = self.scores.compute()
        else:
            self._components = self.components.compute()
            self._scores = self.scores.compute()
    
    def set_attrs(self, attrs: dict):
        '''Set the attributes of the results.'''
        components = self._sanity_check(self._components)
        scores = self._sanity_check(self._scores)
        
        components.attrs.update(attrs)
        scores.attrs.update(attrs)