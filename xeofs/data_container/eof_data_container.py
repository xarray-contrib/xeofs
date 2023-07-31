from typing import Optional

import numpy as np
import xarray as xr
from dask.diagnostics.progress import ProgressBar

from ._base_model_data_container import _BaseModelDataContainer
from ..utils.data_types import DataArray


class EOFDataContainer(_BaseModelDataContainer):
    '''Container to store the results of an EOF analysis.
     
    '''
    def __init__(self):
        super().__init__()
        self._explained_variance: Optional[DataArray] = None
        self._total_variance: Optional[DataArray] = None
        self._idx_modes_sorted: Optional[DataArray] = None

    def set_data(
            self, 
            input_data: DataArray,
            components: DataArray,
            scores: DataArray,
            explained_variance: DataArray,
            total_variance: DataArray,
            idx_modes_sorted: DataArray,
        ):
        super().set_data(
            input_data=input_data,
            components=components,
            scores=scores
        )

        self._verify_dims(explained_variance, ('mode',))
        self._explained_variance = explained_variance
        self._explained_variance.name = 'explained_variance'

        self._total_variance = total_variance
        self._total_variance.name = 'total_variance'

        self._verify_dims(idx_modes_sorted, ('mode',))
        self._idx_modes_sorted = idx_modes_sorted
        self._idx_modes_sorted.name = 'idx_modes_sorted'
    
    @property
    def total_variance(self) -> DataArray:
        '''Get the total variance.'''
        total_var = super()._sanity_check(self._total_variance)
        return total_var

    @property
    def explained_variance(self) -> DataArray:
        '''Get the explained variance.'''
        exp_var = super()._sanity_check(self._explained_variance)
        return exp_var

    @property
    def explained_variance_ratio(self) -> DataArray:
        '''Get the explained variance ratio.'''
        expvar_ratio = self.explained_variance / self.total_variance
        expvar_ratio.name = 'explained_variance_ratio'
        expvar_ratio.attrs.update(self.explained_variance.attrs)
        return expvar_ratio

    @property
    def idx_modes_sorted(self) -> DataArray:
        '''Get the index of the sorted explained variance.'''
        idx_modes_sorted = super()._sanity_check(self._idx_modes_sorted)
        return idx_modes_sorted

    @property
    def singular_values(self) -> DataArray:
        '''Get the explained variance.'''
        svals = np.sqrt((self.input_data.sample.size - 1) * self.explained_variance)
        svals.attrs.update(self.explained_variance.attrs)
        svals.name = 'singular_values'
        return svals

    def compute(self, verbose=False):
        super().compute(verbose)

        if verbose: 
            with ProgressBar():
                self._explained_variance = self.explained_variance.compute()
                self._total_variance = self.total_variance.compute()
                self._idx_modes_sorted = self.idx_modes_sorted.compute()
        else:
            self._explained_variance = self.explained_variance.compute()
            self._total_variance = self.total_variance.compute()
            self._idx_modes_sorted = self.idx_modes_sorted.compute()

    def set_attrs(self, attrs: dict):
        '''Set the attributes of the results.'''
        super().set_attrs(attrs)

        explained_variance = self._sanity_check(self._explained_variance)
        total_variance = self._sanity_check(self._total_variance)
        idx_modes_sorted = self._sanity_check(self._idx_modes_sorted)

        explained_variance.attrs.update(attrs)
        total_variance.attrs.update(attrs)
        idx_modes_sorted.attrs.update(attrs)


class ComplexEOFDataContainer(EOFDataContainer):
    '''Container to store the results of a complex EOF analysis.
    
    '''
    @property
    def components_amplitude(self) -> DataArray:
        '''Get the components amplitude.'''
        comp_abs = abs(self.components)
        comp_abs.name = 'components_amplitude'
        return comp_abs

    @property
    def components_phase(self) -> DataArray:
        '''Get the components phase.'''
        comp_phase = xr.apply_ufunc(
            np.angle, self.components, dask='allowed', keep_attrs=True
        )
        comp_phase.name = 'components_phase'
        return comp_phase

    @property
    def scores_amplitude(self) -> DataArray:
        '''Get the scores amplitude.'''
        score_abs = abs(self.scores)
        score_abs.name = 'scores_amplitude'
        return score_abs

    @property
    def scores_phase(self) -> DataArray:
        '''Get the scores phase.'''
        score_phase = xr.apply_ufunc(
            np.angle, self.scores, dask='allowed', keep_attrs=True
        )
        score_phase.name = 'scores_phase'
        return score_phase
