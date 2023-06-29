import numpy as np
import xarray as xr

from .eof import EOF
from .decomposer import Decomposer
from ..utils.xarray_utils import total_variance
from ..utils.data_types import XarrayData, DataArrayList
from ..utils.xarray_utils import _hilbert_transform_with_padding


class ComplexEOF(EOF):

    def __init__(self, n_modes=10, standardize=False, use_coslat=False, use_weights=False, decay_factor=.2, **kwargs):
        super().__init__(n_modes, standardize, use_coslat, **kwargs)
        self._params.update({'decay_factor':decay_factor})

    def _hilbert_transform(self, data, decay_factor=.2):
       return xr.apply_ufunc(
            _hilbert_transform_with_padding,
            self.data,
            input_core_dims=[['sample', 'feature']],
            output_core_dims=[['sample', 'feature']],
            kwargs={'decay_factor': decay_factor},
        )

    def fit(self, data: XarrayData | DataArrayList, dims, weights=None):
        
        n_modes = self._params['n_modes']
        
        super()._preprocessing(data, dims, weights)
        
        # apply hilbert transform:
        self.data = self._hilbert_transform(self.data, decay_factor=self._params['decay_factor'])

        self._total_variance = total_variance(self.data)

        decomposer = Decomposer(n_components=n_modes)
        decomposer.fit(self.data)

        self._singular_values = decomposer.singular_values_
        self._explained_variance = self._singular_values**2 / (self.data.shape[0] - 1)
        self._explained_variance_ratio = self._explained_variance / self._total_variance
        self._components = decomposer.components_
        self._scores = decomposer.scores_

        self._explained_variance.name = 'explained_variance'
        self._explained_variance_ratio.name = 'explained_variance_ratio'

    def transform(self, data: XarrayData | DataArrayList):
        raise NotImplementedError('ComplexEOF does not support transform method.')

    def components_amplitude(self):
        amplitudes = abs(self._components)
        amplitudes.name = 'components_amplitude'
        return self.stacker.inverse_transform_components(amplitudes)
    
    def components_phase(self):
        phases = np.arctan2(self._components.imag, self._components.real)
        phases.name = 'components_phase'
        return self.stacker.inverse_transform_components(phases)

    def scores_amplitude(self):
        amplitudes = abs(self._scores)
        amplitudes.name = 'scores_amplitude'
        return self.stacker.inverse_transform_scores(amplitudes)
    
    def scores_phase(self):
        phases = np.arctan2(self._scores.imag, self._scores.real)
        phases.name = 'scores_phase'
        return self.stacker.inverse_transform_scores(phases)
    
