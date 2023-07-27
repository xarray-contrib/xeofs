import numpy as np
import xarray as xr
from dask.diagnostics.progress import ProgressBar

from ._base_model_data_container import _BaseModelDataContainer
from ..utils.data_types import DataArray


class EOFDataContainer(_BaseModelDataContainer):
    '''Container that holds the related to an EOF model.
     
    '''
    def __init__(
            self, 
            input_data: DataArray,
            components: DataArray,
            scores: DataArray,
            explained_variance: DataArray,
            total_variance: DataArray,
            idx_modes_sorted: DataArray,
        ):
        super().__init__(
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
    def total_variance(self):
        '''Get the total variance.'''
        return self._total_variance

    @property
    def explained_variance(self):
        '''Get the explained variance.'''
        return self._explained_variance

    @property
    def explained_variance_ratio(self):
        '''Get the explained variance ratio.'''
        expvar_ratio = self.explained_variance / self.total_variance
        expvar_ratio.name = 'explained_variance_ratio'
        return expvar_ratio

    @property
    def idx_modes_sorted(self):
        '''Get the index of the sorted explained variance.'''
        return self._idx_modes_sorted

    @property
    def singular_values(self):
        '''Get the explained variance.'''
        svals = np.sqrt((self.input_data.sample.size - 1) * self.explained_variance)
        svals.attrs.update(self.explained_variance.attrs)
        svals.name = 'singular_values'
        return svals

    def compute(self, verbose=False):
        super().compute(verbose)
        if verbose: 
            with ProgressBar():
                self._explained_variance = self._explained_variance.compute()
                self._total_variance = self._total_variance.compute()
        else:
            self._explained_variance = self._explained_variance.compute()
            self._total_variance = self._total_variance.compute()

    def set_attrs(self, attrs: dict):
        '''Set the attributes of the results.'''
        super().set_attrs(attrs)
        self._explained_variance.attrs.update(attrs)
        self._total_variance.attrs.update(attrs)


class ComplexEOFDataContainer(EOFDataContainer):
    '''Container for complex EOF model data.
    
    '''
    @property
    def components_amplitude(self):
        '''Get the components amplitude.'''
        comp_abs = abs(self._components)
        comp_abs.name = 'components_amplitude'
        return comp_abs

    @property
    def components_phase(self):
        '''Get the components phase.'''
        comp_phase = xr.apply_ufunc(
            np.angle, self._components, dask='allowed', keep_attrs=True
        )
        comp_phase.name = 'components_phase'
        return comp_phase

    @property
    def scores_amplitude(self):
        '''Get the scores amplitude.'''
        score_abs = abs(self._scores)
        score_abs.name = 'scores_amplitude'
        return score_abs

    @property
    def scores_phase(self):
        '''Get the scores phase.'''
        score_phase = xr.apply_ufunc(
            np.angle, self._scores, dask='allowed', keep_attrs=True
        )
        score_phase.name = 'scores_phase'
        return score_phase
