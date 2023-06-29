from abc import ABC, abstractmethod

import numpy as np
import xarray as xr
import scipy as sc


from xeofs.models.scaler import Scaler, ListScaler
from xeofs.models.stacker import DataArrayStacker, DataArrayListStacker, DatasetStacker
from xeofs.models.decomposer import Decomposer, CrossDecomposer
from ..utils.data_types import DataArray, DataArrayList, Dataset, XarrayData
from ..utils.tools import get_dims, compute_total_variance, _hilbert_transform_with_padding
from ..utils.testing import pearson_correlation

class _BaseModel(ABC):
    '''
    Abstract base class for EOF model. 

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
    def __init__(self, n_modes=10, standardize=False, use_coslat=False, use_weights=False, **kwargs):
        self._params = {
            'n_modes': n_modes,
            'standardize': standardize,
            'use_coslat': use_coslat,
            'use_weights': use_weights
        }
        self._scaling_params = {
            'with_std': standardize,
            'with_coslat': use_coslat,
            'with_weights': use_weights
        }
    
    @staticmethod
    def _create_scaler(data: XarrayData | DataArrayList, **kwargs):
        if isinstance(data, (xr.DataArray, xr.Dataset)):
            return Scaler(**kwargs)
        elif isinstance(data, list):
            return ListScaler(**kwargs)
        else:
            raise ValueError(f'Cannot scale data of type: {type(data)}')
    
    @staticmethod
    def _create_stacker(data: XarrayData | DataArrayList, **kwargs):
        if isinstance(data, xr.DataArray):
            return DataArrayStacker(**kwargs)
        elif isinstance(data, list):
            return DataArrayListStacker(**kwargs)
        elif isinstance(data, xr.Dataset):
            return DatasetStacker(**kwargs)
        else:
            raise ValueError(f'Cannot stack data of type: {type(data)}')

    def _preprocessing(self, data, dims, weights=None):
        '''Preprocess the data.
        
        This will scale and stack the data.
        
        Parameters:
        -------------
        data: xr.DataArray or list of xarray.DataArray
            Input data.
        dims: tuple
            Tuple specifying the sample dimensions. The remaining dimensions
            will be treated as feature dimensions.
        weights: xr.DataArray or xr.Dataset or None, default=None
            If specified, the input data will be weighted by this array.
        
        '''
        # Set sample and feature dimensions
        sample_dims, feature_dims = get_dims(data, sample_dims=dims)
        self.dims = {'sample': sample_dims, 'feature': feature_dims}
        
        # Scale the data
        self.scaler = self._create_scaler(data, **self._scaling_params)
        self.scaler.fit(data, sample_dims, feature_dims, weights)  # type: ignore
        data = self.scaler.transform(data)

        # Stack the data
        self.stacker = self._create_stacker(data)
        self.stacker.fit(data, sample_dims, feature_dims)  # type: ignore
        self.data = self.stacker.transform(data)  # type: ignore

    @abstractmethod
    def fit(self, data, dims, weights=None):
        '''
        Abstract method to fit the model.

        Parameters:
        -------------
        data: xr.DataArray or list of xarray.DataArray
            Input data.
        dims: tuple
            Tuple specifying the sample dimensions. The remaining dimensions 
            will be treated as feature dimensions.
        weights: xr.DataArray or xr.Dataset or None, default=None
            If specified, the input data will be weighted by this array.

        '''
        # Here follows the implementation to fit the model
        # Typically you want to start by calling self._preprocessing(data, dims, weights)
        # ATTRIBUTES TO BE DEFINED:
        self._total_variance = None
        self._singular_values = None
        self._explained_variance = None
        self._explained_variance_ratio = None
        self._components = None
        self._scores = None

    @abstractmethod
    def transform(self):
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self):
        raise NotImplementedError

    def singular_values(self):
        '''Return the singular values of the model.

        Returns:
        ----------
        singular_values: DataArray
            Singular values of the fitted model.

        '''
        return self._singular_values
    
    def explained_variance(self):
        '''Return explained variance.'''
        return self._explained_variance
    
    def explained_variance_ratio(self):
        '''Return explained variance ratio.'''
        return self._explained_variance_ratio

    def components(self):
        '''Return the components.
        
        The components in EOF anaylsis are the eigenvectors of the covariance matrix
        (or correlation) matrix. Other names include the principal components or EOFs.

        Returns:
        ----------
        components: DataArray | Dataset | List[DataArray]
            Components of the fitted model.

        '''
        return self.stacker.inverse_transform_components(self._components)  #type: ignore
    
    def scores(self):
        '''Return the scores.
        
        The scores in EOF anaylsis are the projection of the data matrix onto the 
        eigenvectors of the covariance matrix (or correlation) matrix. 
        Other names include the principal component (PC) scores or just PCs.

        Returns:
        ----------
        components: DataArray | Dataset | List[DataArray]
            Scores of the fitted model.

        '''
        return self.stacker.inverse_transform_scores(self._scores)  #type: ignore

    def get_params(self):
        return self._params

    def compute(self):
        '''Computing the model will load and compute Dask arrays.'''

        self._total_variance = self._total_variance.compute()  # type: ignore
        self._singular_values = self._singular_values.compute()   # type: ignore
        self._explained_variance = self._explained_variance.compute()   # type: ignore
        self._explained_variance_ratio = self._explained_variance_ratio.compute()   # type: ignore
        self._components = self._components.compute()    # type: ignore
        self._scores = self._scores.compute()    # type: ignore


class EOF(_BaseModel):
    '''Model to perform Empirical Orthogonal Function (EOF) analysis.
    ComplexEOF
    EOF analysis is more commonly referend to as principal component analysis.

    Parameters:
    -------------
    n_modes: int, default=10
        Number of modes to calculate.
    standardize: bool, default=False
        Whether to standardize the input data.
    use_coslat: bool, default=False
        Whether to use cosine of latitude for scaling.
    
    '''

    def fit(self, data: XarrayData | DataArrayList, dims, weights=None):
        
        n_modes = self._params['n_modes']
        
        super()._preprocessing(data, dims, weights)

        self._total_variance = compute_total_variance(self.data)

        decomposer = Decomposer(n_components=n_modes)
        decomposer.fit(self.data)

        self._singular_values = decomposer.singular_values_
        self._explained_variance = self._singular_values**2 / (self.data.shape[0] - 1)
        self._explained_variance_ratio = self._explained_variance / self._total_variance
        self._components = decomposer.components_
        self._scores = decomposer.scores_    

        self._explained_variance.name = 'explained_variance'
        self._explained_variance_ratio.name = 'explained_variance_ratio'

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
        projections = xr.dot(data, self._components) / self._singular_values
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

        self._total_variance = compute_total_variance(self.data)

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
    

class _CrossBaseModel(ABC):
    '''
    Abstract base class for cross-decomposition models. 

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
    def __init__(self, n_modes=10, standardize=False, use_coslat=False, use_weights=False, **kwargs):
        self._params = {
            'n_modes': n_modes,
            'standardize': standardize,
            'use_coslat': use_coslat,
            'use_weights': use_weights
        }
        self._scaling_params = {
            'with_std': standardize,
            'with_coslat': use_coslat,
            'with_weights': use_weights
        }
    
    @staticmethod
    def _create_scaler(data: XarrayData | DataArrayList, **kwargs):
        if isinstance(data, (xr.DataArray, xr.Dataset)):
            return Scaler(**kwargs)
        elif isinstance(data, list):
            return ListScaler(**kwargs)
        else:
            raise ValueError(f'Cannot scale data of type: {type(data)}')
    
    @staticmethod
    def _create_stacker(data: XarrayData | DataArrayList, **kwargs):
        if isinstance(data, xr.DataArray):
            return DataArrayStacker(**kwargs)
        elif isinstance(data, list):
            return DataArrayListStacker(**kwargs)
        elif isinstance(data, xr.Dataset):
            return DatasetStacker(**kwargs)
        else:
            raise ValueError(f'Cannot stack data of type: {type(data)}')

    def _preprocessing(self, data1, data2, dims, weights1=None, weights2=None):
        '''Preprocess the data.
        
        This will scale and stack the data.
        
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
        # Set sample and feature dimensions
        sample_dims, feature_dims1 = get_dims(data1, sample_dims=dims)
        sample_dims, feature_dims2 = get_dims(data2, sample_dims=dims)
        self.dims = {'sample': sample_dims, 'feature1': feature_dims1, 'feature2': feature_dims2}
        
        # Scale the data
        self.scaler1 = self._create_scaler(data1, **self._scaling_params)
        self.scaler1.fit(data1, sample_dims, feature_dims1, weights1)  # type: ignore
        data1 = self.scaler1.transform(data1)

        self.scaler2 = self._create_scaler(data2, **self._scaling_params)
        self.scaler2.fit(data2, sample_dims, feature_dims2, weights2)  # type: ignore
        data2 = self.scaler2.transform(data2)

        # Stack the data
        self.stacker1 = self._create_stacker(data1)
        self.stacker1.fit(data1, sample_dims, feature_dims1)  # type: ignore
        self.data1 = self.stacker1.transform(data1)  # type: ignore

        self.stacker2 = self._create_stacker(data2)
        self.stacker2.fit(data2, sample_dims, feature_dims2)  # type: ignore
        self.data2 = self.stacker2.transform(data2)  # type: ignore

    @abstractmethod
    def fit(self, data1, data2, dims, weights1=None, weights2=None):
        '''
        Abstract method to fit the model.

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
        # Here follows the implementation to fit the model
        # Typically you want to start by calling self._preprocessing(data1, data2, dims, weights)
        # ATTRIBUTES TO BE DEFINED:
        # self._singular_values
        # self._explained_variance
        # self._squared_total_variance
        # self._squared_covariance_fraction
        # self._singular_vectors1
        # self._singular_vectors2
        # self._norm1
        # self._norm2
        # self._scores1
        # self._scores2

        raise NotImplementedError
    
    @abstractmethod
    def transform(self, data1, data2):
        raise NotImplementedError
    
    @abstractmethod
    def inverse_transform(self, mode):
        raise NotImplementedError

    def get_params(self):
        return self._params
    
    def compute(self):
        '''Computing the model will load and compute Dask arrays.'''

        self._total_variance = self._total_variance.compute()  # type: ignore
        self._singular_values = self._singular_values.compute()   # type: ignore
        self._explained_variance = self._explained_variance.compute()   # type: ignore
        self._explained_variance_ratio = self._explained_variance_ratio.compute()   # type: ignore
        self._components = self._components.compute()    # type: ignore
        self._scores = self._scores.compute()    # type: ignore


class MCA(_CrossBaseModel):

    def fit(self, data1: XarrayData | DataArrayList, data2: XarrayData | DataArrayList, dims, weights1=None, weights2=None):
        '''
        Fit the model.

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

        decomposer = CrossDecomposer(n_components=self._params['n_modes'])
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

    def transform(self, **kwargs):
        '''Project new unseen data onto the singular vectors.

        Parameters:
        -------------
        data1: xr.DataArray or list of xarray.DataArray
            Left input data. Must be provided if `data2` is not provided.
        data2: xr.DataArray or list of xarray.DataArray
            Right input data. Must be provided if `data1` is not provided.

        Returns:
        ----------
        scores1: DataArray | Dataset | List[DataArray]
            Left scores.
        scores2: DataArray | Dataset | List[DataArray]
            Right scores.

        '''
        results = []
        if 'data1' in kwargs:
            data1 = kwargs['data1']
            data1 = self.scaler1.transform(data1)
            data1 = self.stacker1.transform(data1) # type: ignore
            scores1 = xr.dot(data1, self._singular_vectors1) / self._norm1
            scores1 = self.stacker1.inverse_transform_scores(scores1)
            results.append(scores1)

        if 'data2' in kwargs:
            data2 = kwargs['data2']
            data2 = self.scaler2.transform(data2)
            data2 = self.stacker2.transform(data2) # type: ignore
            scores2 = xr.dot(data2, self._singular_vectors2) / self._norm2
            scores2 = self.stacker2.inverse_transform_scores(scores2)
            results.append(scores2)

        return results

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
        Xrec1: DataArray | Dataset | List[DataArray]
            Reconstructed data of left field.
        Xrec2: DataArray | Dataset | List[DataArray]
            Reconstructed data of right field.

        '''
        svecs1 = self._singular_vectors1.sel(mode=mode)  # type: ignore
        svecs2 = self._singular_vectors2.sel(mode=mode)  # type: ignore

        scores1 = self._scores1.sel(mode=mode)  # type: ignore
        scores2 = self._scores2.sel(mode=mode)  # type: ignore

        Xrec1 = svecs1.dot(scores1.conj().T)
        Xrec2 = svecs2.dot(scores2.conj().T)

        Xrec1 = self.stacker1.inverse_transform_data(Xrec1)
        Xrec2 = self.stacker2.inverse_transform_data(Xrec2)

        Xrec1 = self.scaler1.inverse_transform(Xrec1)  # type: ignore
        Xrec2 = self.scaler2.inverse_transform(Xrec2)  # type: ignore

        return Xrec1, Xrec2

    def singular_values(self):
        '''Return the singular values of the model.

        Returns:
        ----------
        singular_values: DataArray
            Singular values of the fitted model.

        '''
        return self._singular_values
    
    def explained_variance(self):
        '''Return explained variance.'''
        return self._explained_variance
    
    def squared_covariance_fraction(self):
        '''Return the squared covariance fraction.
        
        The squared covariance fraction (SCF) is the ratio of the squared covariance
        to the total squared variance and is a measure of importance of the
        mode. It is the analogue of the explained variance ratio in PCA.

        The SCF for mode `i` is given by:

        .. math::
            SCF_i = \\frac{\\sigma_i^2}{\\sum_{i=1}^{n} \\sigma_i^2}

        where :math:`\\sigma_i` is the singular value of mode `i`.

        '''
        return self._squared_covariance_fraction
    
    def components(self):
        '''Return the singular vectors of the left and right field.
        
        Returns:
        ----------
        components1: DataArray | Dataset | List[DataArray]
            Left components of the fitted model.
        components2: DataArray | Dataset | List[DataArray]
            Right components of the fitted model.

        '''
        svecs1 = self.stacker1.inverse_transform_components(self._singular_vectors1)
        svecs2 = self.stacker2.inverse_transform_components(self._singular_vectors2)
        return svecs1, svecs2
    
    def scores(self):
        '''Return the scores of the left and right field.

        The scores in MCA are the projection of the data matrix onto the
        singular vectors of the cross-covariance matrix.
        
        Returns:
        ----------
        scores1: DataArray | Dataset | List[DataArray]
            Left scores.
        scores2: DataArray | Dataset | List[DataArray]
            Right scores.

        '''
        scores1 = self.stacker1.inverse_transform_scores(self._scores1)
        scores2 = self.stacker2.inverse_transform_scores(self._scores2)
        return scores1, scores2

    def homogeneous_patterns(self, correction=None, alpha=0.05):
        '''Return the homogeneous patterns of the left and right field.

        The homogeneous patterns are the correlation coefficients between the 
        input data and the scores.

        More precisely, the homogeneous patterns `r_{hom}` are defined as

        .. math::
          r_{hom, x} = \\corr \\left(X, A_x \\right)
        .. math::
          r_{hom, y} = \\corr \\left(Y, A_y \\right)

        where :math:`X` and :math:`Y` are the input data, :math:`A_x` and :math:`A_y`
        are the scores of the left and right field, respectively.

        Parameters:
        -------------
        correction: str, default=None
            Method to apply a multiple testing correction. If None, no correction
            is applied.  Available methods are:
            - bonferroni : one-step correction
            - sidak : one-step correction
            - holm-sidak : step down method using Sidak adjustments
            - holm : step-down method using Bonferroni adjustments
            - simes-hochberg : step-up method (independent)
            - hommel : closed method based on Simes tests (non-negative)
            - fdr_bh : Benjamini/Hochberg (non-negative) (default)
            - fdr_by : Benjamini/Yekutieli (negative)
            - fdr_tsbh : two stage fdr correction (non-negative)
            - fdr_tsbky : two stage fdr correction (non-negative)
        alpha: float, default=0.05
            The desired family-wise error rate. Not used if `correction` is None.

        Returns:
        ----------
        patterns1: DataArray | Dataset | List[DataArray]
            Left homogenous patterns.
        patterns2: DataArray | Dataset | List[DataArray]
            Right homogenous patterns.
        pvals1: DataArray | Dataset | List[DataArray]
            Left p-values.
        pvals2: DataArray | Dataset | List[DataArray]
            Right p-values.

        '''
        patterns1, pvals1 = pearson_correlation(self.data1, self._scores1, correction=correction, alpha=alpha)
        patterns2, pvals2 = pearson_correlation(self.data2, self._scores2, correction=correction, alpha=alpha)

        patterns1 = self.stacker1.inverse_transform_data(self._singular_vectors1)
        patterns2 = self.stacker2.inverse_transform_data(self._singular_vectors2)

        return patterns1, patterns2, pvals1, pvals2

    def heterogeneous_patterns(self, correction=None, alpha=0.05):
        '''Return the heterogeneous patterns of the left and right field.
        
        The heterogeneous patterns are the correlation coefficients between the
        input data and the scores of the other field.
        
        More precisely, the heterogeneous patterns `r_{het}` are defined as
        
        .. math::
          r_{het, x} = \\corr \\left(X, A_y \\right)
        .. math::
          r_{het, y} = \\corr \\left(Y, A_x \\right)
        
        where :math:`X` and :math:`Y` are the input data, :math:`A_x` and :math:`A_y`
        are the scores of the left and right field, respectively.

        Parameters:
        -------------
        correction: str, default=None
            Method to apply a multiple testing correction. If None, no correction
            is applied.  Available methods are: 
            - bonferroni : one-step correction
            - sidak : one-step correction
            - holm-sidak : step down method using Sidak adjustments
            - holm : step-down method using Bonferroni adjustments
            - simes-hochberg : step-up method (independent)
            - hommel : closed method based on Simes tests (non-negative)
            - fdr_bh : Benjamini/Hochberg (non-negative) (default)
            - fdr_by : Benjamini/Yekutieli (negative)
            - fdr_tsbh : two stage fdr correction (non-negative)
            - fdr_tsbky : two stage fdr correction (non-negative)
        alpha: float, default=0.05
            The desired family-wise error rate. Not used if `correction` is None.

        '''
        patterns1, pvals1 = pearson_correlation(self.data1, self._scores2, correction=correction, alpha=alpha)
        patterns2, pvals2 = pearson_correlation(self.data2, self._scores1, correction=correction, alpha=alpha)

        patterns1 = self.stacker1.inverse_transform_data(self._singular_vectors2)
        patterns2 = self.stacker2.inverse_transform_data(self._singular_vectors1)

        return patterns1, patterns2, pvals1, pvals2

        


class ComplexMCA(MCA):

    def _hilbert_transform(self, data, decay_factor=.2):
       return xr.apply_ufunc(
            _hilbert_transform_with_padding,
            data,
            input_core_dims=[['sample', 'feature']],
            output_core_dims=[['sample', 'feature']],
            kwargs={'decay_factor': decay_factor},
        )

    def fit(self, data1: XarrayData | DataArrayList, data2: XarrayData | DataArrayList, dims, weights1=None, weights2=None):
        '''
        Fit the model.

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
        self.data2 = self._hilbert_transform(self.data1, decay_factor=self._params['decay_factor'])
        self.data1 = self._hilbert_transform(self.data2, decay_factor=self._params['decay_factor'])
        
        decomposer = CrossDecomposer(n_components=self._params['n_modes'])
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

