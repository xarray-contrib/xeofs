from typing import List, Union, Tuple, Dict, Optional, TypeVar, Any, Sequence, Hashable

import numpy as np
import xarray as xr


from ._base_scaler import _BaseScaler
from ..utils.constants import VALID_LATITUDE_NAMES
from ..utils.sanity_checks import assert_dataarray_or_dataset, assert_list_of_dataarrays, ensure_tuple
from ..utils.data_types import DataArray, Dataset, XarrayData, DataArrayList, ModelDims
from ..utils.xarray_utils import sqrt_cos_lat_weights

class Scaler(_BaseScaler):
    '''Scale the data along sample dimensions.
    
    Scaling includes (i) removing the mean and, optionally, (ii) dividing by the standard deviation,
    (iii) multiplying by the square root of cosine of latitude weights (area weighting; coslat weighting), 
    and (iv) multiplying by additional user-defined weights.

    Parameters
    ----------
    with_copy : bool, default=True
        If True, a copy of the data is made before scaling.
    with_std : bool, default=True
        If True, the data is divided by the standard deviation.
    with_coslat : bool, default=False
        If True, the data is multiplied by the square root of cosine of latitude weights.
    with_weights : bool, default=False
        If True, the data is multiplied by additional user-defined weights.

    '''
    
    def fit(self, data: XarrayData, sample_dims: Sequence[Hashable], feature_dims: Sequence[Hashable], weights: Optional[XarrayData]=None):
        '''Fit the scaler to the data.
        
        Parameters
        ----------
        data : xarray.DataArray or xarray.Dataset
            Data to be scaled.
        sample_dims : sequence of hashable
            Dimensions along which the data is considered to be a sample.
        feature_dims : sequence of hashable
            Dimensions along which the data is considered to be a feature.
        weights : xarray.DataArray or xarray.Dataset, optional
            Weights to be applied to the data. Must have the same dimensions as the data.
            If None, no weights are applied.
            
        '''
        # Check input types
        assert_dataarray_or_dataset(data, 'data')
        if weights is not None:
            assert_dataarray_or_dataset(weights, 'weights')

        sample_dims = ensure_tuple(sample_dims)
        feature_dims = ensure_tuple(feature_dims)
        
        # Store sample and feature dimensions for later use
        self.dims: ModelDims = {'sample': sample_dims, 'feature': feature_dims}

        # Scaling parameters are computed along sample dimensions
        self.mean = data.mean(sample_dims).compute()

        if self._params['with_std']:
            self.std = data.std(sample_dims).compute()

        if self._params['with_coslat']:
            self.coslat_weights = self._compute_sqrt_cos_lat_weights(data, feature_dims).compute()
        
        if self._params['with_weights']:
            if weights is None:
                raise ValueError('Weights must be provided when with_weights is True')
            self.weights = weights.compute()


    def transform(self, data: XarrayData) -> XarrayData:
        '''Scale the data.

        Parameters
        ----------
        data : xarray.DataArray or xarray.Dataset
            Data to be scaled.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            Scaled data.

        '''
        assert_dataarray_or_dataset(data, 'data')
        
        if self._params['with_copy']:
            data = data.copy(deep=True)
        
        data = data - self.mean
        
        if self._params['with_std']:
            data = data / self.std
        if self._params['with_coslat']:
            data = data * self.coslat_weights
        if self._params['with_weights']:
            data = data * self.weights
        return data
    
    def inverse_transform(self, data: XarrayData) -> XarrayData:
        '''Unscale the data.

        Parameters
        ----------
        data : xarray.DataArray or xarray.Dataset
            Data to be unscaled.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            Unscaled data.

        '''
        assert_dataarray_or_dataset(data, 'data')
        
        if self._params['with_copy']:
            data = data.copy(deep=True)
        if self._params['with_weights']:
            data = data / self.weights
        if self._params['with_coslat']:
            data = data / self.coslat_weights
        if self._params['with_std']:
            data = data * self.std
        
        data = data + self.mean
        
        return data

    def _compute_sqrt_cos_lat_weights(self, data: XarrayData, dim) -> XarrayData:
        '''Compute the square root of cosine of latitude weights.

        Parameters
        ----------
        data : xarray.DataArray or xarray.Dataset
            Data to be scaled.
        dim : sequence of hashable 
            Dimensions along which the data is considered to be a feature.
            
        Returns
        -------
        xarray.DataArray or xarray.Dataset
            Square root of cosine of latitude weights.

        '''
        assert_dataarray_or_dataset(data, 'data')

        # Find latitude coordinate
        is_lat_coord = np.isin(dim, VALID_LATITUDE_NAMES)

        # Select latitude coordinate and compute coslat weights
        lat_coord = np.array(dim)[is_lat_coord]
        
        if len(lat_coord) > 1:
            raise ValueError(f'{lat_coord} are ambiguous latitude coordinates. Only ONE of the following is allowed for computing coslat weights: {VALID_LATITUDE_NAMES}')

        if len(lat_coord) == 1:
            weights = sqrt_cos_lat_weights(data.coords[lat_coord[0]])
            # Features that cannot be associated to a latitude receive a weight of 1
            weights = weights.where(weights.notnull(), 1)
        else:
            raise ValueError('No latitude coordinate was found to compute coslat weights. Must be one of the following: {:}'.format(VALID_LATITUDE_NAMES))
        weights.name = 'coslat_weights'
        return weights


class ListScaler(_BaseScaler):
    ''' Scale a list of data arrays along sample dimensions.

    Scaling includes (i) removing the mean and, optionally, (ii) dividing by the standard deviation,
    (iii) multiplying by the square root of cosine of latitude weights (area weighting; coslat weighting),
    and (iv) multiplying by additional user-defined weights.

    Parameters
    ----------
    with_copy : bool, default=True
        If True, a copy of the data is made before scaling.
    with_std : bool, default=True
        If True, the data is divided by the standard deviation.
    with_coslat : bool, default=False   
        If True, the data is multiplied by the square root of cosine of latitude weights.
    with_weights : bool, default=False  
        If True, the data is multiplied by additional user-defined weights.
    
    '''
    def __init__(self, with_copy=True, with_std=True, with_coslat=False, with_weights=False):
        super().__init__(with_copy=with_copy, with_std=with_std, with_coslat=with_coslat, with_weights=with_weights)
        self.scalers = []

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
        assert_list_of_dataarrays(data, 'data')

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

            # Create Scaler object for each data array
            params = self.get_params()
            scaler = Scaler(**params)
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
        assert_list_of_dataarrays(da_list, 'da_list')

        da_list_transformed = []
        if self._params['with_copy']:
            da_list = da_list[:]
        for scaler, da in zip(self.scalers, da_list):
            da_list_transformed.append(scaler.transform(da))
        return da_list_transformed
    
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
        assert_list_of_dataarrays(da_list, 'da_list')
        
        da_list_transformed = []
        if self._params['with_copy']:
            da_list = da_list.copy()
        for scaler, da in zip(self.scalers, da_list):
            da_list_transformed.append(scaler.inverse_transform(da))
        return da_list_transformed
    