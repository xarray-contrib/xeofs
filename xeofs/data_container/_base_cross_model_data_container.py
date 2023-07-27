from abc import ABC

from dask.diagnostics.progress import ProgressBar

from ..utils.data_types import DataArray


class _BaseCrossModelDataContainer(ABC):
    '''Abstract base class that holds the cross model data.

    '''
    def __init__(
            self,
            input_data1: DataArray,
            input_data2: DataArray,
            components1: DataArray,
            components2: DataArray,
            scores1: DataArray,
            scores2: DataArray,
        ):
        
        self._verify_dims(input_data1, ('sample', 'feature'))
        self._verify_dims(input_data2, ('sample', 'feature'))
        self._verify_dims(components1, ('feature', 'mode'))
        self._verify_dims(components2, ('feature', 'mode'))
        self._verify_dims(scores1, ('sample', 'mode'))
        self._verify_dims(scores2, ('sample', 'mode'))
        
        components1.name = 'left_components'
        components2.name = 'right_components'
        scores1.name = 'left_scores'
        scores2.name = 'right_scores'

        self._input_data1 = input_data1
        self._input_data2 = input_data2
        self._components1 = components1
        self._components2 = components2
        self._scores1 = scores1
        self._scores2 = scores2

    @staticmethod
    def _verify_dims(da: DataArray, dims: tuple):
        '''Verify that the dimensions of the data are correct.'''
        if not set(da.dims) == set(dims):
            raise ValueError(f'The data must have dimensions {dims} but found {da.dims} instead.')

    @property
    def input_data1(self) -> DataArray:
        '''Get the left input data.'''
        return self._input_data1

    @property
    def input_data2(self) -> DataArray:
        '''Get the right input data.'''
        return self._input_data2

    @property
    def components1(self) -> DataArray:
        '''Get the left components.'''
        return self._components1

    @property
    def components2(self) -> DataArray:
        '''Get the right components.'''
        return self._components2

    @property
    def scores1(self) -> DataArray:
        '''Get the left scores.'''
        return self._scores1

    @property
    def scores2(self) -> DataArray:
        '''Get the right scores.'''
        return self._scores2

    def compute(self, verbose=False):
        '''Compute the results.'''
        if verbose:
            with ProgressBar():
                self._components1 = self._components1.compute()
                self._components2 = self._components2.compute()
                self._scores1 = self._scores1.compute()
                self._scores2 = self._scores2.compute()
        else:
            self._components1 = self._components1.compute()
            self._components2 = self._components2.compute()
            self._scores1 = self._scores1.compute()
            self._scores2 = self._scores2.compute()
    
    def set_attrs(self, attrs: dict):
        '''Set the attributes of the results.'''
        self._components1.attrs.update(attrs)
        self._components2.attrs.update(attrs)
        self._scores1.attrs.update(attrs)
        self._scores2.attrs.update(attrs)