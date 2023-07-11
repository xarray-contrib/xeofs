from typing import List, Union, Tuple, Dict, Optional, TypeVar, Any, Sequence, Hashable

import numpy as np
import xarray as xr


from ._base_scaler import _BaseScaler
from ..utils.constants import VALID_LATITUDE_NAMES
from ..utils.sanity_checks import assert_single_dataarray, assert_single_dataset, assert_list_dataarrays, ensure_tuple
from ..utils.data_types import DataArray, Dataset, DataArrayList, ModelDims, SingleDataObject
from ..utils.xarray_utils import compute_sqrt_cos_lat_weights


class _SingleDataScaler(_BaseScaler):
    '''Scale the data along sample dimensions.
    
    Scaling includes (i) removing the mean and, optionally, (ii) dividing by the standard deviation,
    (iii) multiplying by the square root of cosine of latitude weights (area weighting; coslat weighting), 
    and (iv) multiplying by additional user-defined weights.

    Parameters
    ----------
    with_std : bool, default=True
        If True, the data is divided by the standard deviation.
    with_coslat : bool, default=False
        If True, the data is multiplied by the square root of cosine of latitude weights.
    with_weights : bool, default=False
        If True, the data is multiplied by additional user-defined weights.

    '''
    
    def _verify_input(self, data: SingleDataObject, name: str):
        raise NotImplementedError

    def _compute_sqrt_cos_lat_weights(self, data: SingleDataObject, dim) -> SingleDataObject:
        '''Compute the square root of cosine of latitude weights.

        Parameters
        ----------
        data : SingleDataObject
            Data to be scaled.
        dim : sequence of hashable 
            Dimensions along which the data is considered to be a feature.
            
        Returns
        -------
        SingleDataObject
            Square root of cosine of latitude weights.

        '''
        self._verify_input(data, 'data')

        weights = compute_sqrt_cos_lat_weights(data, dim)
        weights.name = 'coslat_weights'

        return weights

    def fit(self, data: SingleDataObject, sample_dims: Hashable | Sequence[Hashable], feature_dims: Hashable | Sequence[Hashable], weights: Optional[SingleDataObject]=None):
        '''Fit the scaler to the data.
        
        Parameters
        ----------
        data : SingleDataObject
            Data to be scaled.
        sample_dims : sequence of hashable
            Dimensions along which the data is considered to be a sample.
        feature_dims : sequence of hashable
            Dimensions along which the data is considered to be a feature.
        weights : SingleDataObject, optional
            Weights to be applied to the data. Must have the same dimensions as the data.
            If None, no weights are applied.
            
        '''
        # Check input types
        self._verify_input(data, 'data')
        if weights is not None:
            self._verify_input(weights, 'weights')

        sample_dims = ensure_tuple(sample_dims)
        feature_dims = ensure_tuple(feature_dims)
        
        # Store sample and feature dimensions for later use
        self.dims: ModelDims = {'sample': sample_dims, 'feature': feature_dims}

        # Scaling parameters are computed along sample dimensions
        self.mean: SingleDataObject = data.mean(sample_dims).compute()

        if self._params['with_std']:
            self.std: SingleDataObject = data.std(sample_dims).compute()

        if self._params['with_coslat']:
            self.coslat_weights: SingleDataObject = self._compute_sqrt_cos_lat_weights(data, feature_dims).compute()
        
        if self._params['with_weights']:
            if weights is None:
                raise ValueError('Weights must be provided when with_weights is True')
            self.weights: SingleDataObject = weights.compute()


    def transform(self, data: SingleDataObject) -> SingleDataObject:
        '''Scale the data.

        Parameters
        ----------
        data : SingleDataObject
            Data to be scaled.

        Returns
        -------
        SingleDataObject
            Scaled data.

        '''
        self._verify_input(data, 'data')
        
        data = data - self.mean
        
        if self._params['with_std']:
            data = data / self.std
        if self._params['with_coslat']:
            data = data * self.coslat_weights
        if self._params['with_weights']:
            data = data * self.weights
        return data
    
    def fit_transform(self, data: SingleDataObject, sample_dims: Hashable | Sequence[Hashable], feature_dims: Hashable | Sequence[Hashable], weights: Optional[SingleDataObject]=None) -> SingleDataObject:
        '''Fit the scaler to the data and scale it.

        Parameters
        ----------
        data : SingleDataObject
            Data to be scaled.
        sample_dims : sequence of hashable
            Dimensions along which the data is considered to be a sample.
        feature_dims : sequence of hashable
            Dimensions along which the data is considered to be a feature.
        weights : SingleDataObject, optional
            Weights to be applied to the data. Must have the same dimensions as the data.
            If None, no weights are applied.

        Returns
        -------
        SingleDataObject
            Scaled data.

        '''
        self.fit(data, sample_dims, feature_dims, weights)
        return self.transform(data)

    def inverse_transform(self, data: SingleDataObject) -> SingleDataObject:
        '''Unscale the data.

        Parameters
        ----------
        data : SingleDataObject
            Data to be unscaled.

        Returns
        -------
        SingleDataObject
            Unscaled data.

        '''
        self._verify_input(data, 'data')
        
        if self._params['with_weights']:
            data = data / self.weights
        if self._params['with_coslat']:
            data = data / self.coslat_weights
        if self._params['with_std']:
            data = data * self.std
        
        data = data + self.mean
        
        return data


class SingleDataArrayScaler(_SingleDataScaler):

    def _verify_input(self, data: DataArray, name: str):
        '''Verify that the input data is a DataArray.

        Parameters
        ----------
        data : xarray.Dataset
            Data to be checked.

        '''
        assert_single_dataarray(data, name)

    def _compute_sqrt_cos_lat_weights(self, data: DataArray, dim) -> DataArray:
        return super()._compute_sqrt_cos_lat_weights(data, dim)
    
    def fit(
            self,
            data: DataArray,
            sample_dims: Hashable | Sequence[Hashable],
            feature_dims: Hashable | Sequence[Hashable],
            weights: Optional[DataArray]=None
            ):
        super().fit(data, sample_dims, feature_dims, weights)

    def transform(self, data: DataArray) -> DataArray:
        return super().transform(data)
    
    def fit_transform(
            self,
            data: DataArray,
            sample_dims: Hashable | Sequence[Hashable],
            feature_dims: Hashable | Sequence[Hashable],
            weights: Optional[DataArray]=None
        ) -> DataArray:
        return super().fit_transform(data, sample_dims, feature_dims, weights)
    
    def inverse_transform(self, data: DataArray) -> DataArray:
        return super().inverse_transform(data)



class SingleDatasetScaler(_SingleDataScaler):

    def _verify_input(self, data: Dataset, name: str):
        '''Verify that the input data is a Dataset.

        Parameters
        ----------
        data : xarray.Dataset
            Data to be checked.

        '''
        assert_single_dataset(data, name)
    
    def _compute_sqrt_cos_lat_weights(self, data: Dataset, dim) -> Dataset:
        return super()._compute_sqrt_cos_lat_weights(data, dim)
    
    def fit(
            self,
            data: Dataset,
            sample_dims: Hashable | Sequence[Hashable],
            feature_dims: Hashable | Sequence[Hashable],
            weights: Optional[Dataset]=None
            ):
        super().fit(data, sample_dims, feature_dims, weights)

    def transform(self, data: Dataset) -> Dataset:
        return super().transform(data)
    
    def fit_transform(
            self,
            data: Dataset,
            sample_dims: Hashable | Sequence[Hashable],
            feature_dims: Hashable | Sequence[Hashable],
            weights: Optional[Dataset]=None
        ) -> Dataset:
        return super().fit_transform(data, sample_dims, feature_dims, weights)
    
    def inverse_transform(self, data: Dataset) -> Dataset:
        return super().inverse_transform(data)


class ListDataArrayScaler(_BaseScaler):
    ''' Scale a list of xr.DataArray along sample dimensions.

    Scaling includes (i) removing the mean and, optionally, (ii) dividing by the standard deviation,
    (iii) multiplying by the square root of cosine of latitude weights (area weighting; coslat weighting),
    and (iv) multiplying by additional user-defined weights.

    Parameters
    ----------
    with_std : bool, default=True
        If True, the data is divided by the standard deviation.
    with_coslat : bool, default=False   
        If True, the data is multiplied by the square root of cosine of latitude weights.
    with_weights : bool, default=False  
        If True, the data is multiplied by additional user-defined weights.
    
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scalers = []

    def _verify_input(self, data: DataArrayList, name: str):
        '''Verify that the input data is a list of DataArrays.

        Parameters
        ----------
        data : list of xarray.DataArray
            Data to be checked.

        '''
        assert_list_dataarrays(data, name)

    def fit(
            self,
            data: DataArrayList,
            sample_dims: Hashable | Sequence[Hashable],
            feature_dims_list: List[Hashable | Sequence[Hashable]],
            weights=None
            ):
        '''Fit the scaler to the data.

        Parameters
        ----------
        data : list of xarray.DataArray
            Data to be scaled.
        sample_dims : hashable or sequence of hashable
            Dimensions along which the data is considered to be a sample.
        feature_dims_list : list of hashable or list of sequence of hashable
            List of dimensions along which the data is considered to be a feature.
        weights : list of xarray.DataArray, optional
            List of weights to be applied to the data. Must have the same dimensions as the data.

        '''
        self._verify_input(data, 'data')

        # Check input
        if not isinstance(feature_dims_list, list):
            err_message = 'feature dims must be a list of the feature dimensions of each DataArray, '
            err_message += 'e.g. [("lon", "lat"), ("lon")]'
            raise TypeError(err_message)

        sample_dims = ensure_tuple(sample_dims)
        feature_dims = [ensure_tuple(fdims) for fdims in feature_dims_list]

        # Sample dimensions are the same for all data arrays
        # Feature dimensions may be different for each data array
        self.dims : ModelDims = {'sample': sample_dims, 'feature': feature_dims}

        # However, for each DataArray a list of feature dimensions must be provided
        if len(data) != len(feature_dims):
            err_message = 'Number of data arrays and feature dimensions must be the same. '
            err_message += f'Got {len(data)} data arrays and {len(feature_dims)} feature dimensions'
            raise ValueError(err_message)

        self.weights = weights
        # If no weights are provided, create a list of None
        if self.weights is None:
            self.weights = [None] * len(data)
        # Check that number of weights is the same as number of data arrays
        if self._params['with_weights']:
            if len(data) != len(self.weights):
                err_message = 'Number of data arrays and weights must be the same. '
                err_message += f'Got {len(data)} data arrays and {len(self.weights)} weights'
                raise ValueError(err_message)
        
        for da, wghts, fdims in zip(data, self.weights, feature_dims):

            # Create SingleDataArrayScaler object for each data array
            params = self.get_params()
            scaler = SingleDataArrayScaler(**params)
            scaler.fit(da, sample_dims=sample_dims, feature_dims=fdims, weights=wghts)
            self.scalers.append(scaler)

    def transform(self, da_list: DataArrayList) -> DataArrayList:
        '''Scale the data.

        Parameters
        ----------
        da_list : list of xarray.DataArray
            Data to be scaled.

        Returns
        -------
        list of xarray.DataArray
            Scaled data.

        '''
        self._verify_input(da_list, 'da_list')

        da_list_transformed = []
        for scaler, da in zip(self.scalers, da_list):
            da_list_transformed.append(scaler.transform(da))
        return da_list_transformed
    
    def fit_transform(
            self,
            data: DataArrayList,
            sample_dims: Hashable | Sequence[Hashable],
            feature_dims_list: List[Hashable | Sequence[Hashable]],
            weights=None
        ) -> DataArrayList:
        '''Fit the scaler to the data and scale it.

        Parameters
        ----------
        data : list of xr.DataArray
            Data to be scaled.
        sample_dims : hashable or sequence of hashable
            Dimensions along which the data is considered to be a sample.
        feature_dims_list : list of hashable or list of sequence of hashable
            List of dimensions along which the data is considered to be a feature.
        weights : list of xr.DataArray, optional
            List of weights to be applied to the data. Must have the same dimensions as the data.

        Returns
        -------
        list of xarray.DataArray
            Scaled data.

        '''
        self.fit(data, sample_dims, feature_dims_list, weights)
        return self.transform(data)
    
    def inverse_transform(self, da_list: DataArrayList) -> DataArrayList:
        '''Unscale the data.

        Parameters
        ----------
        da_list : list of xarray.DataArray
            Data to be scaled.

        Returns
        -------
        list of xarray.DataArray
            Scaled data.

        '''
        self._verify_input(da_list, 'da_list')
        
        da_list_transformed = []
        for scaler, da in zip(self.scalers, da_list):
            da_list_transformed.append(scaler.inverse_transform(da))
        return da_list_transformed
    