from abc import ABC, abstractmethod

import numpy as np
import xarray as xr
import scipy as sc
from scipy.signal import hilbert    # type: ignore

from xeofs.models.scaler import Scaler, ListScaler
from xeofs.models.stacker import DataArrayStacker, DataArrayListStacker, DatasetStacker
from xeofs.models.decomposer import Decomposer
from ..utils.data_types import DataArray, DataArrayList, Dataset, XarrayData
from ..utils.tools import get_dims, compute_total_variance

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
    
    def _create_scaler(self, data: XarrayData | DataArrayList, **kwargs):
        if isinstance(data, (xr.DataArray, xr.Dataset)):
            return Scaler(**kwargs)
        elif isinstance(data, list):
            return ListScaler(**kwargs)
        else:
            raise ValueError(f'Cannot scale data of type: {type(data)}')
    
    def _create_stacker(self, data: XarrayData | DataArrayList, **kwargs):
        if isinstance(data, xr.DataArray):
            return DataArrayStacker(**kwargs)
        elif isinstance(data, list):
            return DataArrayListStacker(**kwargs)
        elif isinstance(data, xr.Dataset):
            return DatasetStacker(**kwargs)
        else:
            raise ValueError(f'Cannot stack data of type: {type(data)}')

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
    
class ComplexEOF(_BaseModel):

    def __init__(self, n_modes=10, standardize=False, use_coslat=False, use_weights=False, decay_factor=.2, **kwargs):
        super().__init__(n_modes, standardize, use_coslat, **kwargs)
        self._params.update({'decay_factor':decay_factor})

    def _hilbert_transform(self, y, decay_factor=.2):
        n_samples = y.shape[0]

        y = self._pad_exp(y, decay_factor=decay_factor)
        y = hilbert(y, axis=0)
        y = y[n_samples:2*n_samples]
        return y

    def _pad_exp(self, y, decay_factor=.2):
        x = np.arange(y.shape[0])
        x_ext = np.arange(-x.size, 2*x.size)

        coefs = np.polynomial.polynomial.polyfit(x, y, deg=1)
        yfit = np.polynomial.polynomial.polyval(x, coefs).T
        yfit_ext= np.polynomial.polynomial.polyval(x_ext, coefs).T

        y_ano = y - yfit

        amp_pre = y_ano.take(0, axis=0)[:,None]
        amp_pos = y_ano.take(-1, axis=0)[:,None]

        exp_ext = np.exp(-x / x.size / decay_factor)
        exp_ext_reverse = exp_ext[::-1]
        
        pad_pre = amp_pre * exp_ext_reverse
        pad_pos = amp_pos * exp_ext

        y_ext = np.concatenate([pad_pre.T, y_ano, pad_pos.T], axis=0)
        y_ext += yfit_ext
        return y_ext

    def fit(self, data: XarrayData | DataArrayList, dims, weights=None):
        
        n_modes = self._params['n_modes']
        
        super()._preprocessing(data, dims, weights)
        # apply hilbert transform:
        self.data = xr.apply_ufunc(
            self._hilbert_transform,
            self.data,
            input_core_dims=[['sample', 'feature']],
            output_core_dims=[['sample', 'feature']],
            kwargs={'decay_factor': self._params['decay_factor']},
        )

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