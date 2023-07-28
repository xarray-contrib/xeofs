import numpy as np
import xarray as xr

from ._base_model import _BaseModel
from .decomposer import Decomposer
from ..utils.data_types import AnyDataObject, DataArray
from ..data_container.eof_data_container import EOFDataContainer, ComplexEOFDataContainer
from ..utils.xarray_utils import hilbert_transform
from ..utils.xarray_utils import total_variance as compute_total_variance


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

    def fit(self, data: AnyDataObject, dim, weights=None):
        
        n_modes = self._params['n_modes']
        
        input_data: DataArray = self.preprocessor.fit_transform(data, dim, weights)

        # Compute the total variance
        total_variance = compute_total_variance(input_data, dim='sample')

        # Decompose the data
        decomposer = Decomposer(n_modes=n_modes)
        decomposer.fit(input_data)

        # Compute the explained variance
        svals = decomposer.singular_values_
        explained_variance = svals**2 / (input_data.sample.size - 1)

        # Index of the sorted explained variance
        # It's already sorted, we just need to assign it to the DataContainer
        # for the sake of consistency
        idx_modes_sorted = explained_variance.compute().argsort()[::-1]
        idx_modes_sorted.coords.update(explained_variance.coords)

        # Assign the results to the data container
        self.data = EOFDataContainer(
            input_data=input_data,
            components=decomposer.components_,
            scores=decomposer.scores_,
            explained_variance=explained_variance,
            total_variance=total_variance,
            idx_modes_sorted=idx_modes_sorted,
        )
        self.data.set_attrs(self.attrs)

    def transform(self, data: AnyDataObject) -> DataArray:
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
        data_stacked: DataArray = self.preprocessor.transform(data)

        components = self.data.components
        singular_values = self.data.singular_values

        # Project the data
        projections = xr.dot(data_stacked, components, dims='feature') / singular_values
        projections.name = 'scores'

        # Unstack the projections
        projections = self.preprocessor.inverse_transform_scores(projections)
        return projections
    
    def inverse_transform(self, mode) -> AnyDataObject:
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
        svals = self.data.singular_values.sel(mode=mode)
        comps = self.data.components.sel(mode=mode)
        scores = self.data.scores.sel(mode=mode) * svals
        reconstructed_data = xr.dot(comps.conj(), scores)
        reconstructed_data.name = 'reconstructed_data'

        # Enforce real output
        reconstructed_data = reconstructed_data.real

        # Unstack and unscale the data
        reconstructed_data = self.preprocessor.inverse_transform_data(reconstructed_data)
        return reconstructed_data

    def singular_values(self):
        '''Return the singular values of the model.

        Returns:
        ----------
        singular_values: DataArray
            Singular values of the fitted model.

        '''
        return self.data.singular_values
    
    def explained_variance(self):
        '''Return explained variance.'''
        return self.data.explained_variance
    
    def explained_variance_ratio(self):
        '''Return explained variance ratio.'''
        return self.data.explained_variance_ratio


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

    def fit(self, data: AnyDataObject, dim, weights=None):
        
        n_modes = self._params['n_modes']
        
        input_data: DataArray = self.preprocessor.fit_transform(data, dim, weights)
        
        # apply hilbert transform:
        padding = self._params['padding']
        decay_factor = self._params['decay_factor']
        input_data = hilbert_transform(
            input_data, dim='sample',
            padding=padding, decay_factor=decay_factor
        )

        # Compute the total variance
        total_variance = compute_total_variance(input_data, dim='sample')
        
        # Decompose the complex data
        decomposer = Decomposer(n_modes=n_modes)
        decomposer.fit(input_data)

        # Compute the explained variance
        svals = decomposer.singular_values_
        explained_variance = svals**2 / (input_data.sample.size - 1)

        # Index of the sorted explained variance
        # It's already sorted, we just need to assign it to the DataContainer
        # for the sake of consistency
        idx_modes_sorted = explained_variance.compute().argsort()[::-1]
        idx_modes_sorted.coords.update(explained_variance.coords)

        self.data = ComplexEOFDataContainer(
            input_data=input_data,
            components=decomposer.components_,
            scores=decomposer.scores_,
            explained_variance=explained_variance,
            total_variance=total_variance,
            idx_modes_sorted=idx_modes_sorted,
        )
        # Assign analysis-relevant meta data to the results
        self.data.set_attrs(self.attrs)

    def transform(self, data: AnyDataObject):
        raise NotImplementedError('ComplexEOF does not support transform method.')

    def components_amplitude(self) -> AnyDataObject:
        amplitudes = self.data.components_amplitude
        return self.preprocessor.inverse_transform_components(amplitudes)
    
    def components_phase(self) -> AnyDataObject:
        phases = self.data.components_phase
        return self.preprocessor.inverse_transform_components(phases)

    def scores_amplitude(self) -> AnyDataObject:
        amplitudes = self.data.scores_amplitude
        return self.preprocessor.inverse_transform_scores(amplitudes)
    
    def scores_phase(self) -> AnyDataObject:
        phases = self.data.scores_phase
        return self.preprocessor.inverse_transform_scores(phases)
    
