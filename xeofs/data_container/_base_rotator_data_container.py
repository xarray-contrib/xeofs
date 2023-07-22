from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np
import xarray as xr
from dask.diagnostics.progress import ProgressBar

from ._base_model_data_container import _BaseModelDataContainer
from ..utils.data_types import DataArray
from ..models.eof import EOF, ComplexEOF

Model = TypeVar('Model', EOF, ComplexEOF)


class _BaseRotatorDataContainer(_BaseModelDataContainer):
    '''Abstract base class for rotator results.

    '''
    def __init__(
            self,
            input_data: DataArray,
            components: DataArray,
            scores: DataArray,
            rotation_matrix: DataArray,
            phi_matrix: DataArray,
        ):
        super().__init__(
            input_data=input_data,
            components=components,
            scores=scores
        )
        super()._verify_dims(rotation_matrix, ('mode', 'mode1'))
        super()._verify_dims(phi_matrix, ('mode', 'mode1'))
        
        self._rotation_matrx = rotation_matrix
        self._phi_matrix = phi_matrix

        self._rotation_matrx.name = 'rotation_matrix'
        self._phi_matrix.name = 'phi_matrix'

    @property
    def rotation_matrix(self) -> DataArray:
        '''Get the rotation matrix.'''
        return self._rotation_matrx
    
    @property
    def phi_matrix(self) -> DataArray:
        '''Get the phi matrix.'''
        return self._phi_matrix

    def compute(self, verbose=False):
        '''Compute the results.'''
        super().compute(verbose=verbose)
        if verbose:
            with ProgressBar():
                self._rotation_matrx = self._rotation_matrx.compute()
                self._phi_matrix = self._phi_matrix.compute()
        else:
            self._rotation_matrx = self._rotation_matrx.compute()
            self._phi_matrix = self._phi_matrix.compute()
    
    def set_attrs(self, attrs: dict):
        '''Set the attributes of the results.'''
        super().set_attrs(attrs)
        self._rotation_matrx.attrs.update(attrs)
        self._phi_matrix.attrs.update(attrs)
