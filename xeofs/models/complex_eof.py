import numpy as np
import xarray as xr

from .eof import EOF
from .decomposer import Decomposer
from ..utils.xarray_utils import total_variance
from ..utils.data_types import XarrayData, DataArrayList
from ..utils.xarray_utils import hilbert_transform


class ComplexEOF(EOF):
    '''Decomposes a data object using Complex Empirical Orthogonal Functions (EOF).
    
    Complex EOFs are computed by applying a Hilbert transform to the data before
    computing the EOFs. The Hilbert transform is applied to each feature of the
    data individually. Optionally, the Hilbert transform is applied after padding
    the data with exponentially decaying values to mitigate the impact of spectral leakage.

    Parameters
    ----------
    n_modes : int
        Number of modes to be computed.
    standardize : bool
        If True, standardize the data before computing the EOFs.
    use_coslat : bool
        If True, weight the data by the square root of the cosine of the latitude
        weights.
    use_weights : bool
        If True, weight the data by the weights.
    padding : Optional, str
        Padding method for the Hilbert transform to mitigate spectral leakage. Currently, only ``'exp'`` is
        supported.
    decay_factor : float
        Decay factor of the exponential padding. Only used if ``padding='exp'``. A good value typically
        depends on the data. If the data is highly variable, a small value (e.g. 0.05) is recommended. For
        data with low variability, a larger value (e.g. 0.2) is recommended.

    '''

    def __init__(self, n_modes=10, standardize=False, use_coslat=False, use_weights=False, padding='exp', decay_factor=.2, **kwargs):
        super().__init__(n_modes, standardize, use_coslat, use_weights, **kwargs)
        self._hilbert_params = {'padding': padding, 'decay_factor': decay_factor}

    def fit(self, data: XarrayData | DataArrayList, dims, weights=None):
        
        n_modes = self._params['n_modes']
        
        super()._preprocessing(data, dims, weights)
        
        # apply hilbert transform:
        self.data = hilbert_transform(self.data, dim='sample', **self._hilbert_params)

        self._total_variance = total_variance(self.data)

        decomposer = Decomposer(n_modes=n_modes)
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
    
