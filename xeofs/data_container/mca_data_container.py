import numpy as np
import xarray as xr
from dask.diagnostics.progress import ProgressBar

from ._base_cross_model_data_container import _BaseCrossModelDataContainer
from ..utils.data_types import DataArray
from ..utils.statistics import pearson_correlation


class MCADataContainer(_BaseCrossModelDataContainer):
    '''Container that holds the related to an MCA model.
     
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
            norm1: DataArray,
            norm2: DataArray,
        ):
        super().__init__(
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
    def total_squared_covariance(self):
        '''Get the total squared covariance.'''
        return self._total_squared_covariance

    @property
    def squared_covariance(self):
        '''Get the squared covariance.'''
        return self._squared_covariance

    @property
    def squared_covariance_fraction(self):
        '''Get the squared covariance fraction (SCF).'''
        scf = self.squared_covariance / self.total_squared_covariance
        scf.attrs.update(self._squared_covariance.attrs)
        scf.name = 'squared_covariance_fraction'
        return scf

    @property
    def norm1(self):
        '''Get the norm of the left scores.'''
        return self._norm1
    
    @property
    def norm2(self):
        '''Get the norm of the right scores.'''
        return self._norm2
    
    @property
    def idx_modes_sorted(self):
        '''Get the indices of the modes sorted by the squared covariance.'''
        return self._idx_modes_sorted
    
    def get_homogeneous_patterns(self, alpha=0.05, correction=None):
        '''Get the homogeneous patterns.'''
        hom_pat1, pvals1 = pearson_correlation(self.input_data1, self.scores1, alpha=alpha, correction=correction)
        hom_pat2, pvals2 = pearson_correlation(self.input_data2, self.scores2, alpha=alpha, correction=correction)

        hom_pat1.name = 'left_homogeneous_pattern'
        hom_pat2.name = 'right_homogeneous_pattern'
        pvals1.name = 'left_homogeneous_pattern_pvalues'
        pvals2.name = 'right_homogeneous_pattern_pvalues'
        return (hom_pat1, hom_pat2), (pvals1, pvals2)
    
    def get_heterogeneous_patterns(self, alpha=0.05, correction=None):
        '''Get the heterogeneous patterns.'''
        het_pat1, pvals1 = pearson_correlation(self.input_data1, self.scores2, alpha=alpha, correction=correction)
        het_pat2, pvals2 = pearson_correlation(self.input_data2, self.scores1, alpha=alpha, correction=correction)

        het_pat1.name = 'left_heterogeneous_pattern'
        het_pat2.name = 'right_heterogeneous_pattern'
        pvals1.name = 'left_heterogeneous_pattern_pvalues'
        pvals2.name = 'right_heterogeneous_pattern_pvalues'
        return (het_pat1, het_pat2), (pvals1, pvals2)

    def compute(self, verbose=False):
        super().compute(verbose)
        if verbose:
            with ProgressBar():
                self._total_squared_covariance = self._total_squared_covariance.compute()
                self._squared_covariance = self._squared_covariance.compute()
                self._norm1 = self._norm1.compute()
                self._norm2 = self._norm2.compute()
        else:
            self._total_squared_covariance = self._total_squared_covariance.compute()
            self._squared_covariance = self._squared_covariance.compute()
            self._norm1 = self._norm1.compute()
            self._norm2 = self._norm2.compute()

    def set_attrs(self, attrs: dict):
        super().set_attrs(attrs)
        self._total_squared_covariance.attrs.update(attrs)
        self._squared_covariance.attrs.update(attrs)
        self._norm1.attrs.update(attrs)
        self._norm2.attrs.update(attrs)


class ComplexMCADataContainer(MCADataContainer):
    '''Container that holds the data related to a Complex MCA model.
     
    '''
    def get_homogeneous_patterns(self, alpha=0.05, correction=None):
        raise NotImplementedError('Not defined.')
    
    def get_heterogeneous_patterns(self, alpha=0.05, correction=None):
        raise NotImplementedError('Not defined.')
    
    @property
    def components_amplitude1(self):
        '''Get the component amplitudes of the left field.'''
        comp_amps1 = abs(self._components1)
        comp_amps1.name = 'left_components_amplitude'
        return comp_amps1
    
    @property
    def components_amplitude2(self):
        '''Get the component amplitudes of the right field.'''
        comp_amps2 = abs(self._components2)
        comp_amps2.name = 'right_components_amplitude'
        return comp_amps2
    
    @property
    def components_phase1(self):
        '''Get the component phases of the left field.'''
        comp_phs1 = xr.apply_ufunc(np.angle, self._components1, keep_attrs=True)
        comp_phs1.name = 'left_components_phase'
        return comp_phs1
    
    @property
    def components_phase2(self):
        '''Get the component phases of the right field.'''
        comp_phs2 = xr.apply_ufunc(np.angle, self._components2, keep_attrs=True)
        comp_phs2.name = 'right_components_phase'
        return comp_phs2
    
    @property
    def scores_amplitude1(self):
        '''Get the scores amplitudes of the left field.'''
        scores_amps1 = abs(self._scores1)
        scores_amps1.name = 'left_scores_amplitude'
        return scores_amps1
    
    @property
    def scores_amplitude2(self):
        '''Get the scores amplitudes of the right field.'''
        scores_amps2 = abs(self._scores2)
        scores_amps2.name = 'right_scores_amplitude'
        return scores_amps2
    
    @property
    def scores_phase1(self):
        '''Get the scores phases of the left field.'''
        scores_phs1 = xr.apply_ufunc(np.angle, self._scores1, keep_attrs=True)
        scores_phs1.name = 'left_scores_phase'
        return scores_phs1
    
    @property
    def scores_phase2(self):
        '''Get the scores phases of the right field.'''
        scores_phs2 = xr.apply_ufunc(np.angle, self._scores2, keep_attrs=True)
        scores_phs2.name = 'right_scores_phase'
        return scores_phs2