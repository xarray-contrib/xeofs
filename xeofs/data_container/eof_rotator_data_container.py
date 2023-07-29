from abc import abstractmethod
from typing import TypeVar, Optional

import numpy as np
import xarray as xr
from dask.diagnostics.progress import ProgressBar

from .eof_data_container import EOFDataContainer, ComplexEOFDataContainer
from ..utils.data_types import DataArray


class EOFRotatorDataContainer(EOFDataContainer):
    '''Container to store the results of a rotated EOF analysis.
     
    '''
    def __init__(self):
        super().__init__()
        self._rotation_matrix: Optional[DataArray] = None
        self._phi_matrix: Optional[DataArray] = None
        self._modes_sign: Optional[DataArray] = None

    def set_data(
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
        super().set_data(
            input_data=input_data,
            components=components,
            scores=scores,
            explained_variance=explained_variance,
            total_variance=total_variance,
            idx_modes_sorted=idx_modes_sorted,
        )
        
        self._verify_dims(rotation_matrix, ('mode_m', 'mode_n'))
        self._rotation_matrix = rotation_matrix
        self._rotation_matrix.name = 'rotation_matrix'

        self._verify_dims(phi_matrix, ('mode_m', 'mode_n'))
        self._phi_matrix = phi_matrix
        self._phi_matrix.name = 'phi_matrix'

        self._verify_dims(modes_sign, ('mode',))
        self._modes_sign = modes_sign
        self._modes_sign.name = 'modes_sign'


    @property
    def rotation_matrix(self) -> DataArray:
        '''Get the rotation matrix.'''
        rotation_matrix = super()._sanity_check(self._rotation_matrix)
        return rotation_matrix
    
    @property
    def phi_matrix(self) -> DataArray:
        '''Get the phi matrix.'''
        phi_matrix = super()._sanity_check(self._phi_matrix)
        return phi_matrix
    
    @property
    def modes_sign(self) -> DataArray:
        '''Get the modes sign.'''
        modes_sign = super()._sanity_check(self._modes_sign)
        return modes_sign

    def compute(self, verbose:bool=False):
        super().compute(verbose)

        if verbose:
            with ProgressBar():
                self._rotation_matrix = self.rotation_matrix.compute()
                self._phi_matrix = self.phi_matrix.compute()
                self._modes_sign = self.modes_sign.compute()
        else:
            self._rotation_matrix = self.rotation_matrix.compute()
            self._phi_matrix = self.phi_matrix.compute()
            self._modes_sign = self.modes_sign.compute()

    def set_attrs(self, attrs: dict):
        super().set_attrs(attrs)
        self.rotation_matrix.attrs.update(attrs)
        self.phi_matrix.attrs.update(attrs)
        self.modes_sign.attrs.update(attrs)



class ComplexEOFRotatorDataContainer(EOFRotatorDataContainer, ComplexEOFDataContainer):
    '''Container to store the results of a complex rotated EOF analysis.
     
    '''
    def __init__(self):
        super(ComplexEOFRotatorDataContainer, self).__init__()

    def set_data(
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
        super().set_data(
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

