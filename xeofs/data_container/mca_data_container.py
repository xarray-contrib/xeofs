from typing import Optional

import numpy as np
import xarray as xr
from dask.diagnostics.progress import ProgressBar

from ._base_cross_model_data_container import _BaseCrossModelDataContainer
from ..utils.data_types import DataArray


class MCADataContainer(_BaseCrossModelDataContainer):
    '''Container to store the results of a MCA.
     
    '''
    def __init__(self):
        super().__init__()
        self._squared_covariance: Optional[DataArray] = None
        self._total_squared_covariance: Optional[DataArray] = None
        self._idx_modes_sorted: Optional[DataArray] = None
        self._norm1: Optional[DataArray] = None
        self._norm2: Optional[DataArray] = None

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
            norm1: DataArray,
            norm2: DataArray,
        ):
        super().set_data(
            input_data1=input_data1,
            input_data2=input_data2,
            components1=components1,
            components2=components2,
            scores1=scores1,
            scores2=scores2,
        )

        self._verify_dims(squared_covariance, ('mode',))
        self._squared_covariance = squared_covariance
        self._squared_covariance.name = 'squared_covariance'

        self._total_squared_covariance = total_squared_covariance
        self._total_squared_covariance.name = 'total_squared_covariance'

        self._verify_dims(idx_modes_sorted, ('mode',))
        self._idx_modes_sorted = idx_modes_sorted
        self._idx_modes_sorted.name = 'idx_modes_sorted'

        self._verify_dims(norm1, ('mode',))
        self._norm1 = norm1
        self._norm1.name = 'left_norm'

        self._verify_dims(norm2, ('mode',))
        self._norm2 = norm2
        self._norm2.name = 'right_norm'
    
    @property
    def total_squared_covariance(self) -> DataArray:
        '''Get the total squared covariance.'''
        tsc = super()._sanity_check(self._total_squared_covariance)
        return tsc

    @property
    def squared_covariance(self) -> DataArray:
        '''Get the squared covariance.'''
        sc = super()._sanity_check(self._squared_covariance)
        return sc

    @property
    def squared_covariance_fraction(self) -> DataArray:
        '''Get the squared covariance fraction (SCF).'''
        scf = self.squared_covariance / self.total_squared_covariance
        scf.attrs.update(self.squared_covariance.attrs)
        scf.name = 'squared_covariance_fraction'
        return scf

    @property
    def norm1(self) -> DataArray:
        '''Get the norm of the left scores.'''
        norm1 = super()._sanity_check(self._norm1)
        return norm1
    
    @property
    def norm2(self) -> DataArray:
        '''Get the norm of the right scores.'''
        norm2 = super()._sanity_check(self._norm2)
        return norm2
    
    @property
    def idx_modes_sorted(self) -> DataArray:
        '''Get the indices of the modes sorted by the squared covariance.'''
        idx_modes_sorted = super()._sanity_check(self._idx_modes_sorted)
        return idx_modes_sorted
    
    @property
    def singular_values(self) -> DataArray:
        '''Get the singular values.'''
        singular_values = xr.apply_ufunc(np.sqrt, self.squared_covariance, dask='allowed', vectorize=False, keep_attrs=True)
        singular_values.name = 'singular_values'
        return singular_values
    
    @property
    def total_covariance(self) -> DataArray:
        '''Get the total covariance.
        
        This measure follows the defintion of Cheng and Dunkerton (1995).
        Note that this measure is not an invariant in MCA.
        
        '''
        tot_cov = self.singular_values.sum()
        tot_cov.attrs.update(self.singular_values.attrs)
        tot_cov.name = 'total_covariance'
        return tot_cov
    
    @property
    def covariance_fraction(self) -> DataArray:
        '''Get the covariance fraction (CF).
        
        This measure follows the defintion of Cheng and Dunkerton (1995).
        Note that this measure is not an invariant in MCA.
        
        '''
        cov_frac = self.singular_values / self.total_covariance
        cov_frac.attrs.update(self.singular_values.attrs)
        cov_frac.name = 'covariance_fraction'
        return cov_frac

    def compute(self, verbose=False):
        super().compute(verbose)

        if verbose:
            with ProgressBar():
                self._total_squared_covariance = self.total_squared_covariance.compute()
                self._squared_covariance = self.squared_covariance.compute()
                self._norm1 = self.norm1.compute()
                self._norm2 = self.norm2.compute()
        else:
            self._total_squared_covariance = self.total_squared_covariance.compute()
            self._squared_covariance = self.squared_covariance.compute()
            self._norm1 = self.norm1.compute()
            self._norm2 = self.norm2.compute()

    def set_attrs(self, attrs: dict):
        super().set_attrs(attrs)

        total_squared_covariance = super()._sanity_check(self._total_squared_covariance)
        squared_covariance = super()._sanity_check(self._squared_covariance)
        norm1 = super()._sanity_check(self._norm1)
        norm2 = super()._sanity_check(self._norm2)

        total_squared_covariance.attrs.update(attrs)
        squared_covariance.attrs.update(attrs)
        norm1.attrs.update(attrs)
        norm2.attrs.update(attrs)


class ComplexMCADataContainer(MCADataContainer):
    '''Container that holds the data related to a Complex MCA model.
     
    '''
    @property
    def components_amplitude1(self) -> DataArray:
        '''Get the component amplitudes of the left field.'''
        comp_amps1 = abs(self.components1)
        comp_amps1.name = 'left_components_amplitude'
        return comp_amps1
    
    @property
    def components_amplitude2(self) -> DataArray:
        '''Get the component amplitudes of the right field.'''
        comp_amps2 = abs(self.components2)
        comp_amps2.name = 'right_components_amplitude'
        return comp_amps2
    
    @property
    def components_phase1(self) -> DataArray:
        '''Get the component phases of the left field.'''
        comp_phs1 = xr.apply_ufunc(np.angle, self.components1, keep_attrs=True)
        comp_phs1.name = 'left_components_phase'
        return comp_phs1
    
    @property
    def components_phase2(self) -> DataArray:
        '''Get the component phases of the right field.'''
        comp_phs2 = xr.apply_ufunc(np.angle, self._components2, keep_attrs=True)
        comp_phs2.name = 'right_components_phase'
        return comp_phs2
    
    @property
    def scores_amplitude1(self) -> DataArray:
        '''Get the scores amplitudes of the left field.'''
        scores_amps1 = abs(self.scores1)
        scores_amps1.name = 'left_scores_amplitude'
        return scores_amps1
    
    @property
    def scores_amplitude2(self) -> DataArray:
        '''Get the scores amplitudes of the right field.'''
        scores_amps2 = abs(self.scores2)
        scores_amps2.name = 'right_scores_amplitude'
        return scores_amps2
    
    @property
    def scores_phase1(self) -> DataArray:
        '''Get the scores phases of the left field.'''
        scores_phs1 = xr.apply_ufunc(np.angle, self.scores1, keep_attrs=True)
        scores_phs1.name = 'left_scores_phase'
        return scores_phs1
    
    @property
    def scores_phase2(self) -> DataArray:
        '''Get the scores phases of the right field.'''
        scores_phs2 = xr.apply_ufunc(np.angle, self.scores2, keep_attrs=True)
        scores_phs2.name = 'right_scores_phase'
        return scores_phs2