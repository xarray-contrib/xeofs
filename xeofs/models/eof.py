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

    Parameters
    ----------
    n_modes: int, default=10
        Number of modes to calculate.
    standardize: bool, default=False
        Whether to standardize the input data.
    use_coslat: bool, default=False
        Whether to use cosine of latitude for scaling.
    use_weights: bool, default=False
        Whether to use weights.
    
    Examples
    --------
    >>> model = xe.models.EOF(n_modes=5)
    >>> model.fit(data)
    >>> scores = model.scores()

    '''
    def __init__(self, n_modes=10, standardize=False, use_coslat=False, use_weights=False):
        super().__init__(n_modes=n_modes, standardize=standardize, use_coslat=use_coslat, use_weights=use_weights)
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

        Parameters
        ----------
        data: AnyDataObject
            Data to be transformed.

        Returns
        -------
        projections: DataArray
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

        Parameters
        ----------
        mode: integer, a list of integers, or a slice object.
            The mode(s) used to reconstruct the data. If a scalar is given,
            the data will be reconstructed using the given mode. If a slice
            is given, the data will be reconstructed using the modes in the
            given slice. If a list of integers is given, the data will be reconstructed
            using the modes in the given list.

        Returns
        -------
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
        '''Return the (EOF) components.
        
        The components in EOF anaylsis are the eigenvectors of the covariance/correlation matrix.
        Other names include the principal components or EOFs.

        Returns
        -------
        components: DataArray | Dataset | List[DataArray]
            Components of the fitted model.

        '''
        components = self.data.components
        return self.preprocessor.inverse_transform_components(components)
    
    def scores(self) -> DataArray:
        '''Return the (PC) scores.
        
        The scores in EOF anaylsis are the projection of the data matrix onto the 
        eigenvectors of the covariance matrix (or correlation) matrix. 
        Other names include the principal component (PC) scores or just PCs.

        Returns
        -------
        components: DataArray | Dataset | List[DataArray]
            Scores of the fitted model.

        '''
        scores = self.data.scores
        return self.preprocessor.inverse_transform_scores(scores)

    def singular_values(self) -> DataArray:
        '''Return the singular values of the Singular Value Decomposition.

        Returns
        -------
        singular_values: DataArray
            Singular values obtained from the SVD.

        '''
        return self.data.singular_values
    
    def explained_variance(self) -> DataArray:
        '''Return explained variance.
        
        The explained variance :math:`\\lambda_i` is the variance explained 
        by each mode. It is defined as

        .. math::
            \\lambda_i = \\frac{\\sigma_i^2}{N-1}

        where :math:`\\sigma_i` is the singular value of the :math:`i`-th mode and :math:`N` is the number of samples.
        Equivalently, :math:`\\lambda_i` is the :math:`i`-th eigenvalue of the covariance matrix.

        Returns
        -------
        explained_variance: DataArray
            Explained variance.
        '''
        return self.data.explained_variance
    
    def explained_variance_ratio(self) -> DataArray:
        '''Return explained variance ratio.
        
        The explained variance ratio :math:`\\gamma_i` is the variance explained
        by each mode normalized by the total variance. It is defined as

        .. math::
            \\gamma_i = \\frac{\\lambda_i}{\\sum_{j=1}^M \\lambda_j}

        where :math:`\\lambda_i` is the explained variance of the :math:`i`-th mode and :math:`M` is the total number of modes.

        Returns
        -------
        explained_variance_ratio: DataArray
            Explained variance ratio.
        '''
        return self.data.explained_variance_ratio


class ComplexEOF(EOF):
    '''Complex Empirical Orthogonal Functions (Complex EOF) analysis.

    The Complex EOF analysis [1]_ [2]_ (also known as Hilbert EOF analysis) applies a Hilbert transform 
    to the data before performing the standard EOF analysis. 
    The Hilbert transform is applied to each feature of the data individually.

    An optional padding with exponentially decaying values can be applied prior to
    the Hilbert transform in order to mitigate the impact of spectral leakage.

    Parameters
    ----------
    n_modes : int
        Number of modes to calculate.
    standardize : bool
        Whether to standardize the input data.
    use_coslat : bool
        Whether to use cosine of latitude for scaling.
    use_weights : bool
        Whether to use weights.
    padding : str, optional
        Specifies the method used for padding the data prior to applying the Hilbert
        transform. This can help to mitigate the effect of spectral leakage. 
        Currently, only 'exp' for exponential padding is supported. Default is 'exp'.
    decay_factor : float, optional
        Specifies the decay factor used in the exponential padding. This parameter
        is only used if padding='exp'. The recommended value typically ranges between 0.05 to 0.2 
        but ultimately depends on the variability in the data. 
        A smaller value (e.g. 0.05) is recommended for
        data with high variability, while a larger value (e.g. 0.2) is recommended
        for data with low variability. Default is 0.2.

    References
    ----------
    .. [1] Horel, J., 1984. Complex Principal Component Analysis: Theory and Examples. J. Climate Appl. Meteor. 23, 1660–1673. https://doi.org/10.1175/1520-0450(1984)023<1660:CPCATA>2.0.CO;2
    .. [2] Hannachi, A., Jolliffe, I., Stephenson, D., 2007. Empirical orthogonal functions and related techniques in atmospheric science: A review. International Journal of Climatology 27, 1119–1152. https://doi.org/10.1002/joc.1499

    Examples
    --------
    >>> model = ComplexEOF(n_modes=5, standardize=True)
    >>> model.fit(data)
        
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
        '''Return the amplitude of the (EOF) components.
        
        The amplitude of the components are defined as

        .. math::
            A_ij = |C_ij|

        where :math:`C_{ij}` is the :math:`i`-th entry of the :math:`j`-th component and
        :math:`|\\cdot|` denotes the absolute value.

        Returns
        -------
        components_amplitude: DataArray | Dataset | List[DataArray]
            Amplitude of the components of the fitted model.

        '''
        amplitudes = self.data.components_amplitude
        return self.preprocessor.inverse_transform_components(amplitudes)
    
    def components_phase(self) -> AnyDataObject:
        '''Return the phase of the (EOF) components.

        The phase of the components are defined as

        .. math::
            \\phi_{ij} = \\arg(C_{ij})

        where :math:`C_{ij}` is the :math:`i`-th entry of the :math:`j`-th component and
        :math:`\\arg(\\cdot)` denotes the argument of a complex number.

        Returns
        -------
        components_phase: DataArray | Dataset | List[DataArray]
            Phase of the components of the fitted model.

        '''
        phases = self.data.components_phase
        return self.preprocessor.inverse_transform_components(phases)

    def scores_amplitude(self) -> DataArray:
        '''Return the amplitude of the (PC) scores.

        The amplitude of the scores are defined as

        .. math::
            A_ij = |S_ij|

        where :math:`S_{ij}` is the :math:`i`-th entry of the :math:`j`-th score and
        :math:`|\\cdot|` denotes the absolute value.

        Returns
        -------
        scores_amplitude: DataArray | Dataset | List[DataArray]
            Amplitude of the scores of the fitted model.

        '''
        amplitudes = self.data.scores_amplitude
        return self.preprocessor.inverse_transform_scores(amplitudes)
    
    def scores_phase(self) -> DataArray:
        '''Return the phase of the (PC) scores.

        The phase of the scores are defined as
        
        .. math::
            \\phi_{ij} = \\arg(S_{ij})

        where :math:`S_{ij}` is the :math:`i`-th entry of the :math:`j`-th score and
        :math:`\\arg(\\cdot)` denotes the argument of a complex number.

        Returns
        -------
        scores_phase: DataArray | Dataset | List[DataArray]
            Phase of the scores of the fitted model.

        '''
        phases = self.data.scores_phase
        return self.preprocessor.inverse_transform_scores(phases)
    
