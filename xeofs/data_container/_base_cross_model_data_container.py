from abc import ABC
from typing import Optional

from dask.diagnostics.progress import ProgressBar

from ..utils.data_types import DataArray


class _BaseCrossModelDataContainer(ABC):
    '''Abstract base class that holds the cross model data.

    '''
    def __init__(self):
        self._input_data1: Optional[DataArray] = None
        self._input_data2: Optional[DataArray] = None
        self._components1: Optional[DataArray] = None
        self._components2: Optional[DataArray] = None
        self._scores1: Optional[DataArray] = None
        self._scores2: Optional[DataArray] = None

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

    def set_data(
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


    @property
    def input_data1(self) -> DataArray:
        '''Get the left input data.'''
        data1 = self._sanity_check(self._input_data1)
        return data1

    @property
    def input_data2(self) -> DataArray:
        '''Get the right input data.'''
        data2 = self._sanity_check(self._input_data2)
        return data2

    @property
    def components1(self) -> DataArray:
        '''Get the left components.'''
        components1 = self._sanity_check(self._components1)
        return components1

    @property
    def components2(self) -> DataArray:
        '''Get the right components.'''
        components2 = self._sanity_check(self._components2)
        return components2

    @property
    def scores1(self) -> DataArray:
        '''Get the left scores.'''
        scores1 = self._sanity_check(self._scores1)
        return scores1

    @property
    def scores2(self) -> DataArray:
        '''Get the right scores.'''
        scores2 = self._sanity_check(self._scores2)
        return scores2

    def compute(self, verbose=False):
        '''Compute and load delayed dask DataArrays into memory.
        
        Parameters
        ----------
        verbose : bool
            Whether or not to provide additional information about the computing progress.
        '''
        if verbose:
            with ProgressBar():
                self._components1 = self.components1.compute()
                self._components2 = self.components2.compute()
                self._scores1 = self.scores1.compute()
                self._scores2 = self.scores2.compute()
        else:
            self._components1 = self.components1.compute()
            self._components2 = self.components2.compute()
            self._scores1 = self.scores1.compute()
            self._scores2 = self.scores2.compute()
    
    def set_attrs(self, attrs: dict):
        '''Set the attributes of the results.'''
        components1 = self._sanity_check(self._components1)
        components2 = self._sanity_check(self._components2)
        scores1 = self._sanity_check(self._scores1)
        scores2 = self._sanity_check(self._scores2)

        components1.attrs.update(attrs)
        components2.attrs.update(attrs)
        scores1.attrs.update(attrs)
        scores2.attrs.update(attrs)