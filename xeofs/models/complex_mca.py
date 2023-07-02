import numpy as np
import xarray as xr

from .mca import MCA
from .decomposer import CrossDecomposer
from ..utils.data_types import XarrayData, DataArrayList
from ..utils.xarray_utils import hilbert_transform


class ComplexMCA(MCA):
    '''
    A class used to perform complex maximum covariance analysis (MCA) on two sets of data. 

    This class inherits from the MCA class and overloads its methods to implement a version of MCA 
    that uses complex numbers (i.e., applies the Hilbert transform) to capture phase relationships 
    in the input datasets.

    ...

    Attributes
    ----------
    No additional attributes to the MCA base class.

    Methods
    -------
    fit(data1, data2, dims, weights1=None, weights2=None):
        Fit the model to two datasets.

    transform(data1, data2):
        Not implemented in the ComplexMCA class.

    homogeneous_patterns(correction=None, alpha=0.05):
        Not implemented in the ComplexMCA class.

    heterogeneous_patterns(correction=None, alpha=0.05):
        Not implemented in the ComplexMCA class.
    '''

    def __init__(self, n_modes=10, standardize=False, use_coslat=False, use_weights=False, padding='exp', decay_factor=.2, **kwargs):
        super().__init__(n_modes=n_modes, standardize=standardize, use_coslat=use_coslat, use_weights=use_weights, **kwargs)
        self._hilbert_params = {'padding': padding, 'decay_factor': decay_factor}

    def fit(self, data1: XarrayData | DataArrayList, data2: XarrayData | DataArrayList, dims, weights1=None, weights2=None):
        '''Fit the model.

        Parameters:
        -------------
        data1: xr.DataArray or list of xarray.DataArray
            Left input data.
        data2: xr.DataArray or list of xarray.DataArray
            Right input data.
        dims: tuple
            Tuple specifying the sample dimensions. The remaining dimensions 
            will be treated as feature dimensions.
        weights1: xr.DataArray or xr.Dataset or None, default=None
            If specified, the left input data will be weighted by this array.
        weights2: xr.DataArray or xr.Dataset or None, default=None
            If specified, the right input data will be weighted by this array.

        '''

        self._preprocessing(data1, data2, dims, weights1, weights2)
        
        # apply hilbert transform:
        self.data1 = hilbert_transform(self.data1, dim='sample', **self._hilbert_params)
        self.data2 = hilbert_transform(self.data2, dim='sample', **self._hilbert_params)
        
        decomposer = CrossDecomposer(n_modes=self._params['n_modes'])
        decomposer.fit(self.data1, self.data2)

        # Note:
        # - explained variance is given by the singular values of the SVD;
        # - We use the term singular_values_pca as used in the context of PCA:
        # Considering data X1 = X2, MCA is the same as PCA. In this case,
        # singular_values_pca is equivalent to the singular values obtained
        # when performing PCA of X1 or X2.
        self._singular_values = decomposer.singular_values_
        self._explained_variance = decomposer.singular_values_  # actually covariance
        self._squared_total_variance = decomposer.squared_total_variance_
        self._squared_covariance_fraction = self._explained_variance**2 / self._squared_total_variance
        self._singular_values_pca = np.sqrt(self._singular_values * (self.data1.shape[0] - 1))
        self._singular_vectors1 = decomposer.singular_vectors1_
        self._singular_vectors2 = decomposer.singular_vectors2_
        self._norm1 = np.sqrt(self._singular_values)
        self._norm2 = np.sqrt(self._singular_values)

        self._explained_variance.name = 'explained_variance'
        self._squared_total_variance.name = 'squared_total_variance'
        self._squared_covariance_fraction.name = 'squared_covariance_fraction'
        self._norm1.name = 'left_norm'
        self._norm2.name = 'right_norm'

        # Project the data onto the singular vectors
        sqrt_expvar = np.sqrt(self._explained_variance)
        self._scores1 = xr.dot(self.data1, self._singular_vectors1) / sqrt_expvar
        self._scores2 = xr.dot(self.data2, self._singular_vectors2) / sqrt_expvar

    
    def transform(self, data1: XarrayData | DataArrayList, data2: XarrayData | DataArrayList):
        raise NotImplementedError('Complex MCA does not support transform method.')

    def homogeneous_patterns(self, correction=None, alpha=0.05):
        raise NotImplementedError('Complex MCA does not support homogeneous_patterns method.')
    
    def heterogeneous_patterns(self, correction=None, alpha=0.05):
        raise NotImplementedError('Complex MCA does not support heterogeneous_patterns method.')

