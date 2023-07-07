import numpy as np
import xarray as xr

from ._base_model import _BaseModel
from .decomposer import Decomposer
from ..utils.data_types import XarrayData, DataArrayList
from ..utils.xarray_utils import total_variance
from ..utils.xarray_utils import hilbert_transform


class EOF(_BaseModel):
    '''Empirical Orthogonal Functions (EOF) analysis.

    EOF analysis is more commonly referend to as principal component analysis (PCA).

    Parameters:
    -------------
    n_modes: int, default=10
        Number of modes to calculate.
    standardize: bool, default=False
        Whether to standardize the input data.
    use_coslat: bool, default=False
        Whether to use cosine of latitude for scaling.
    use_weights: bool, default=False
        Whether to use weights.
    
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attrs.update({'model': 'EOF analysis'})

    def fit(self, data: XarrayData | DataArrayList, dim, weights=None):
        
        n_modes = self._params['n_modes']
        
        super()._preprocessing(data, dim, weights)

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

        # Assign analysis relevant meta data
        self._assign_meta_data()

    def transform(self, data: XarrayData | DataArrayList) -> XarrayData | DataArrayList:
        '''Project new unseen data onto the components (EOFs/eigenvectors).

        Parameters:
        -------------
        data: xr.DataArray or list of xarray.DataArray
            Input data.

        Returns:
        ----------
        projections: DataArray | Dataset | List[DataArray]
            Projections of the new data onto the components.

        '''
        # Preprocess the data
        data = self.scaler.transform(data)  #type: ignore
        data = self.stacker.transform(data)  #type: ignore

        # Project the data
        projections = xr.dot(data, self._components, dims='feature') / self._singular_values
        projections.name = 'scores'

        # Unstack the projections
        projections = self.stacker.inverse_transform_scores(projections)
        return projections
    
    def inverse_transform(self, mode):
        '''Reconstruct the original data from transformed data.

        Parameters:
        -------------
        mode: scalars, slices or array of tick labels.
            The mode(s) used to reconstruct the data. If a scalar is given,
            the data will be reconstructed using the given mode. If a slice
            is given, the data will be reconstructed using the modes in the
            given slice. If a array is given, the data will be reconstructed
            using the modes in the given array.

        Returns:
        ----------
        data: DataArray | Dataset | List[DataArray]
            Reconstructed data.

        '''
        # Reconstruct the data
        svals = self._singular_values.sel(mode=mode)  # type: ignore
        comps = self._components.sel(mode=mode)  # type: ignore
        scores = self._scores.sel(mode=mode) * svals  # type: ignore
        data = xr.dot(comps, scores)
        data.name = 'reconstructed_data'

        # Unstack the data
        data = self.stacker.inverse_transform_data(data)
        # Unscale the data
        data = self.scaler.inverse_transform(data)  # type: ignore
        return data





class ComplexEOF(EOF):
    '''Complex Empirical Orthogonal Functions (EOF) analysis.
    
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

    def __init__(self, padding='exp', decay_factor=.2, **kwargs):
        super().__init__(**kwargs)
        self._name = 'Complex EOF analysis'    
        self._params.update({'padding': padding, 'decay_factor': decay_factor})

    def fit(self, data: XarrayData | DataArrayList, dim, weights=None):
        
        n_modes = self._params['n_modes']
        
        super()._preprocessing(data, dim, weights)
        
        # apply hilbert transform:
        padding = self._params['padding']
        decay_factor = self._params['decay_factor']
        self.data = hilbert_transform(
            self.data, dim='sample',
            padding=padding, decay_factor=decay_factor
        )

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

        # Assign analysis-relevant meta data to the results
        self._assign_meta_data()

    def transform(self, data: XarrayData | DataArrayList):
        raise NotImplementedError('ComplexEOF does not support transform method.')

    def components_amplitude(self):
        amplitudes = abs(self._components)
        amplitudes.name = 'components_amplitude'
        return self.stacker.inverse_transform_components(amplitudes)
    
    def components_phase(self):
        phases = xr.apply_ufunc(np.angle, self._components, dask='allowed', keep_attrs=True)
        phases.name = 'components_phase'
        return self.stacker.inverse_transform_components(phases)

    def scores_amplitude(self):
        amplitudes = abs(self._scores)
        amplitudes.name = 'scores_amplitude'
        return self.stacker.inverse_transform_scores(amplitudes)
    
    def scores_phase(self):
        phases = xr.apply_ufunc(np.angle, self._scores, dask='allowed', keep_attrs=True)
        phases.name = 'scores_phase'
        return self.stacker.inverse_transform_scores(phases)
    
