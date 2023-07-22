from abc import abstractmethod
from typing import TypeVar

import numpy as np
import xarray as xr
from dask.diagnostics.progress import ProgressBar

from .eof_data_container import EOFDataContainer, ComplexEOFDataContainer
from ..utils.data_types import DataArray
from ..models.eof import EOF, ComplexEOF
from ..utils.xarray_utils import total_variance as compute_total_variance

Model = TypeVar('Model', EOF, ComplexEOF)


class EOFRotatorDataContainer(EOFDataContainer):
    '''Container for rotated EOF model data.
     
    '''
    def __init__(
            self, 
            input_data: DataArray,
            components: DataArray,
            scores: DataArray,
            explained_variance: DataArray,
            total_variance: DataArray,
            idx_modes_sorted: DataArray,
            modes_sign: DataArray,
            rotation_matrix: DataArray,
            phi_matrix: DataArray,
        ):
        super().__init__(
            input_data=input_data,
            components=components,
            scores=scores,
            explained_variance=explained_variance,
            total_variance=total_variance,
            idx_modes_sorted=idx_modes_sorted,
        )
        
        self._verify_dims(rotation_matrix, ('mode', 'mode1'))
        self._rotation_matrix = rotation_matrix
        self._rotation_matrix.name = 'rotation_matrix'

        self._verify_dims(phi_matrix, ('mode', 'mode1'))
        self._phi_matrix = phi_matrix
        self._phi_matrix.name = 'phi_matrix'

        self._verify_dims(modes_sign, ('mode',))
        self._modes_sign = modes_sign
        self._modes_sign.name = 'modes_sign'


    @property
    def rotation_matrix(self) -> DataArray:
        '''Get the rotation matrix.'''
        return self._rotation_matrix
    
    @property
    def phi_matrix(self) -> DataArray:
        '''Get the phi matrix.'''
        return self._phi_matrix
    
    @property
    def modes_sign(self) -> DataArray:
        '''Get the modes sign.'''
        return self._modes_sign

    def compute(self, verbose:bool=False):
        '''Compute and load delayed dask objects into memory.
        
        Parameters
        ----------
        verbose : bool
            Whether or not to provide additional information about the computing progress.
        '''
        # Compute rotated solution
        super().compute(verbose)
        if verbose:
            with ProgressBar():
                self._rotation_matrix = self._rotation_matrix.compute()
                self._phi_matrix = self._phi_matrix.compute()
                self._modes_sign = self._modes_sign.compute()
        else:
            self._rotation_matrix = self._rotation_matrix.compute()
            self._phi_matrix = self._phi_matrix.compute()
            self._modes_sign = self._modes_sign.compute()

    def set_attrs(self, attrs: dict):
        super().set_attrs(attrs)
        self._rotation_matrix.attrs.update(attrs)
        self._phi_matrix.attrs.update(attrs)
        self._modes_sign.attrs.update(attrs)



class ComplexEOFRotatorDataContainer(EOFRotatorDataContainer, ComplexEOFDataContainer):
    '''Container for rotated EOF model data.
     
    '''

    def __init__(
            self, 
            input_data: DataArray,
            components: DataArray,
            scores: DataArray,
            explained_variance: DataArray,
            total_variance: DataArray,
            idx_modes_sorted: DataArray,
            rotation_matrix: DataArray,
            phi_matrix: DataArray,
            modes_sign: DataArray,
        ):
        super().__init__(
            input_data=input_data,
            components=components,
            scores=scores,
            explained_variance=explained_variance,
            total_variance=total_variance,
            idx_modes_sorted=idx_modes_sorted,
            rotation_matrix=rotation_matrix,
            phi_matrix=phi_matrix,
            modes_sign=modes_sign,
        )

