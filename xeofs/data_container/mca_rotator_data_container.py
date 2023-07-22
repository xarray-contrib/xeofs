import numpy as np
import xarray as xr
from dask.diagnostics.progress import ProgressBar

from xeofs.utils.data_types import DataArray

from .mca_data_container import MCADataContainer, ComplexMCADataContainer
from ..utils.data_types import DataArray


class MCARotatorDataContainer(MCADataContainer):
    '''Container that holds the data related to a rotated MCA model.
     
    '''

    def __init__(
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
        super().__init__(
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
    def rotation_matrix(self):
        '''Get the rotation matrix.'''
        return self._rotation_matrix
    
    @property
    def phi_matrix(self):
        '''Get the phi matrix.'''
        return self._phi_matrix
    
    @property
    def modes_sign(self):
        '''Get the mode signs.'''
        return self._modes_sign
    
    def compute(self, verbose: bool = False):
        '''Compute the rotated MCA model.
        
        '''
        super().compute(verbose=verbose)

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
        '''Set the attributes of the data container.
            
        '''
        super().set_attrs(attrs)

        self._rotation_matrix.attrs.update(attrs)
        self._phi_matrix.attrs.update(attrs)
        self._modes_sign.attrs.update(attrs)


class ComplexMCARotatorDataContainer(MCARotatorDataContainer, ComplexMCADataContainer):
    '''Container that holds the data related to a rotated complex MCA model.
     
    '''

    def __init__(
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
        super().__init__(
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

