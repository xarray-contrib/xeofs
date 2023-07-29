import xarray as xr

from ._base_model import _BaseModel
from .decomposer import Decomposer
from ..utils.data_types import AnyDataObject, DataArray
from ..data_container import EOFDataContainer, ComplexEOFDataContainer
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

        # Initialize the DataContainer to store the results
        self.data: EOFDataContainer = EOFDataContainer()


    def fit(self, data: AnyDataObject, dim, weights=None):
        # Preprocess the data
        input_data: DataArray = self.preprocessor.fit_transform(data, dim, weights)

        # Compute the total variance
        total_variance = compute_total_variance(input_data, dim='sample')

        # Decompose the data
        n_modes = self._params['n_modes']

        decomposer = Decomposer(n_modes=n_modes)
        decomposer.fit(input_data)

        singular_values = decomposer.singular_values_
        components = decomposer.components_
        scores = decomposer.scores_

        # Compute the explained variance
        explained_variance = singular_values**2 / (input_data.sample.size - 1)

        # Index of the sorted explained variance
        # It's already sorted, we just need to assign it to the DataContainer
        # for the sake of consistency
        idx_modes_sorted = explained_variance.compute().argsort()[::-1]
        idx_modes_sorted.coords.update(explained_variance.coords)

        # Assign the results to the data container
        self.data.set_data(
            input_data=input_data,
            components=components,
            scores=scores,
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

    def components(self) -> AnyDataObject:
        '''Return the components.
        
        The components in EOF anaylsis are the eigenvectors of the covariance matrix
        (or correlation) matrix. Other names include the principal components or EOFs.

        Returns:
        ----------
        components: DataArray | Dataset | List[DataArray]
            Components of the fitted model.

        '''
        components = self.data.components
        return self.preprocessor.inverse_transform_components(components)
    
    def scores(self) -> DataArray:
        '''Return the scores.
        
        The scores in EOF anaylsis are the projection of the data matrix onto the 
        eigenvectors of the covariance matrix (or correlation) matrix. 
        Other names include the principal component (PC) scores or just PCs.

        Returns:
        ----------
        components: DataArray | Dataset | List[DataArray]
            Scores of the fitted model.

        '''
        scores = self.data.scores
        return self.preprocessor.inverse_transform_scores(scores)

    def singular_values(self) -> DataArray:
        '''Return the singular values of the model.

        Returns:
        ----------
        singular_values: DataArray
            Singular values of the fitted model.

        '''
        return self.data.singular_values
    
    def explained_variance(self) -> DataArray:
        '''Return explained variance.'''
        return self.data.explained_variance
    
    def explained_variance_ratio(self) -> DataArray:
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
        self.attrs.update({'model': 'Complex EOF analysis'}) 
        self._params.update({'padding': padding, 'decay_factor': decay_factor})

        # Initialize the DataContainer to store the results
        self.data: ComplexEOFDataContainer = ComplexEOFDataContainer()

    def fit(self, data: AnyDataObject, dim, weights=None):        
        # Preprocess the data
        input_data: DataArray = self.preprocessor.fit_transform(data, dim, weights)
        
        # Apply hilbert transform:
        padding = self._params['padding']
        decay_factor = self._params['decay_factor']
        input_data = hilbert_transform(
            input_data, dim='sample',
            padding=padding, decay_factor=decay_factor
        )

        # Compute the total variance
        total_variance = compute_total_variance(input_data, dim='sample')
        
        # Decompose the complex data
        n_modes = self._params['n_modes']

        decomposer = Decomposer(n_modes=n_modes)
        decomposer.fit(input_data)

        singular_values = decomposer.singular_values_
        components = decomposer.components_
        scores = decomposer.scores_

        # Compute the explained variance
        explained_variance = singular_values**2 / (input_data.sample.size - 1)

        # Index of the sorted explained variance
        # It's already sorted, we just need to assign it to the DataContainer
        # for the sake of consistency
        idx_modes_sorted = explained_variance.compute().argsort()[::-1]
        idx_modes_sorted.coords.update(explained_variance.coords)

        self.data.set_data(
            input_data=input_data,
            components=components,
            scores=scores,
            explained_variance=explained_variance,
            total_variance=total_variance,
            idx_modes_sorted=idx_modes_sorted,
        )
        # Assign analysis-relevant meta data to the results
        self.data.set_attrs(self.attrs)

    def transform(self, data: AnyDataObject) -> DataArray:
        raise NotImplementedError('ComplexEOF does not support transform method.')

    def components_amplitude(self) -> AnyDataObject:
        amplitudes = self.data.components_amplitude
        return self.preprocessor.inverse_transform_components(amplitudes)
    
    def components_phase(self) -> AnyDataObject:
        phases = self.data.components_phase
        return self.preprocessor.inverse_transform_components(phases)

    def scores_amplitude(self) -> DataArray:
        amplitudes = self.data.scores_amplitude
        return self.preprocessor.inverse_transform_scores(amplitudes)
    
    def scores_phase(self) -> DataArray:
        phases = self.data.scores_phase
        return self.preprocessor.inverse_transform_scores(phases)
    
