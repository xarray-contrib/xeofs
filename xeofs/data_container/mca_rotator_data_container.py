from typing import Optional

import numpy as np
import xarray as xr
from dask.diagnostics.progress import ProgressBar

from xeofs.utils.data_types import DataArray

from .mca_data_container import MCADataContainer, ComplexMCADataContainer
from ..utils.data_types import DataArray


class MCARotatorDataContainer(MCADataContainer):
    '''Container that holds the data related to a rotated MCA model.
     
    '''
    def __init__(self):
        super().__init__()
        self._rotation_matrix: Optional[DataArray] = None
        self._phi_matrix: Optional[DataArray] = None
        self._modes_sign: Optional[DataArray] = None


    def set_data(
            self,
            input_data1: DataArray,
            input_data2: DataArray,
            components1: DataArray,
            components2: DataArray,
            scores1: DataArray,
            scores2: DataArray,
            squared_covariance: DataArray,
            total_squared_covariance: DataArray,
            idx_modes_sorted: DataArray,
            modes_sign: DataArray,
            norm1: DataArray,
            norm2: DataArray,
            rotation_matrix: DataArray,
            phi_matrix: DataArray,
        ):
        super().set_data(
            input_data1=input_data1,
            input_data2=input_data2,
            components1=components1,
            components2=components2,
            scores1=scores1,
            scores2=scores2,
            squared_covariance=squared_covariance,
            total_squared_covariance=total_squared_covariance,
            idx_modes_sorted=idx_modes_sorted,
            norm1=norm1,
            norm2=norm2,
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
        '''Get the mode signs.'''
        modes_sign = super()._sanity_check(self._modes_sign)
        return modes_sign
    
    def compute(self, verbose: bool = False):
        '''Compute the rotated MCA model.
        
        '''
        super().compute(verbose=verbose)

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
        '''Set the attributes of the data container.
            
        '''
        super().set_attrs(attrs)

        rotation_matrix = super()._sanity_check(self._rotation_matrix)
        phi_matrix = super()._sanity_check(self._phi_matrix)
        modes_sign = super()._sanity_check(self._modes_sign)

        rotation_matrix.attrs.update(attrs)
        phi_matrix.attrs.update(attrs)
        modes_sign.attrs.update(attrs)


class ComplexMCARotatorDataContainer(MCARotatorDataContainer, ComplexMCADataContainer):
    '''Container that holds the data related to a rotated complex MCA model.
     
    '''
    def __init__(self):
        super(ComplexMCARotatorDataContainer, self).__init__()

    def set_data(
            self,
            input_data1: DataArray,
            input_data2: DataArray,
            components1: DataArray,
            components2: DataArray,
            scores1: DataArray,
            scores2: DataArray,
            squared_covariance: DataArray,
            total_squared_covariance: DataArray,
            idx_modes_sorted: DataArray,
            modes_sign: DataArray,
            norm1: DataArray,
            norm2: DataArray,
            rotation_matrix: DataArray,
            phi_matrix: DataArray,
        ):
        super().set_data(
            input_data1=input_data1,
            input_data2=input_data2,
            components1=components1,
            components2=components2,
            scores1=scores1,
            scores2=scores2,
            squared_covariance=squared_covariance,
            total_squared_covariance=total_squared_covariance,
            idx_modes_sorted=idx_modes_sorted,
            modes_sign=modes_sign,
            norm1=norm1,
            norm2=norm2,
            rotation_matrix=rotation_matrix,
            phi_matrix=phi_matrix,
        )

