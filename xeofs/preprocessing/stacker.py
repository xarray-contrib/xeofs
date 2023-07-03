from typing import List, Sequence, Hashable

import numpy as np
import xarray as xr

from xeofs.utils.data_types import DataArray

from ._base_stacker import _BaseStacker
from ..utils.data_types import DataArray, DataArrayList, Dataset
from ..utils.sanity_checks import ensure_tuple


class DataArrayStacker(_BaseStacker):
    ''' Converts a DataArray of any dimensionality into a 2D structure.

    This operation generates a reshaped DataArray with two distinct dimensions: 'sample' and 'feature'.
    
    The handling of NaNs is specific: if they are found to populate an entire dimension (be it 'sample' or 'feature'), 
    they are temporarily removed during transformations and subsequently reinstated. 
    However, the presence of isolated NaNs will trigger an error.
        
    '''

    def _to_2d(self, data: DataArray, sample_dims, feature_dims) -> DataArray:
        ''' Reshape a DataArray to 2D.

        Parameters
        ----------
        data : DataArray
            The data to be reshaped.
        sample_dims : Hashable or Sequence[Hashable]
            The dimensions of the data that will be stacked along the `sample` dimension.
        feature_dims : Hashable or Sequence[Hashable]
            The dimensions of the data that will be stacked along the `feature` dimension.

        Returns
        -------
        data_stacked : DataArray
            The reshaped 2d-data.
        '''
        return data.stack(sample=sample_dims, feature=feature_dims)

    def _reindex_dim(self, data: DataArray,  model_dim : str):
        ''' Reindex data to original coordinates in case that some features at the boundaries were dropped
        
        Parameters
        ----------
        data : DataArray
            The data to be reindex.
        model_dim : str ['sample', 'feature']
            The dimension to be reindexed.
            
        Returns
        -------
        DataArray
            The reindexed data.
            
        '''
        # check if coordinates in self.coords have different length from data.coords
        # if so, reindex data.coords to self.coords
        # input_dim : dimensions of input data
        # model_dim : dimensions of model data i.e. sample or feature
        for input_dim in self.coords[model_dim].keys():
            if self.coords[model_dim][input_dim].size != data.coords[input_dim].size:
                data = data.reindex({input_dim: self.coords[model_dim][input_dim]})
        return data

    def fit(
            self,
            data: DataArray,
            sample_dims: Hashable | Sequence[Hashable],
            feature_dims: Hashable | Sequence[Hashable]
            ):
        ''' Fit the stacker to the data.
        
        Parameters
        ----------
        data : DataArray
            The data to be reshaped.
        sample_dims : Hashable or Sequence[Hashable]
            The dimensions of the data that will be stacked along the `sample` dimension.
        feature_dims : Hashable or Sequence[Hashable]
            The dimensions of the data that will be stacked along the `feature` dimension.

        '''

        sample_dims = ensure_tuple(sample_dims)
        feature_dims = ensure_tuple(feature_dims)
        self.dims = {'sample': sample_dims, 'feature': feature_dims}
        self.coords = {
            'sample': {dim: data.coords[dim] for dim in sample_dims},
            'feature': {dim: data.coords[dim] for dim in feature_dims}
        }

    def transform(self, data: DataArray) -> DataArray:
        ''' Reshape the data into a 2D version.
        
        Parameters
        ----------
        data : DataArray
            The data to be reshaped.
            
        Returns
        -------
        DataArray
            The reshaped data.

        Raises
        ------
        ValueError
            If the data to be transformed has different dimensions than the data used to fit the stacker.
        ValueError
            If the data to be transformed has different coordinates than the data used to fit the stacker.
        ValueError
            If the data to be transformed has individual NaNs.
            
        '''
        # Test whether sample and feature dimensions are present in data array
        dim_samples_exist = np.isin(self.dims['sample'], np.array(data.dims)).all()
        dim_features_exist = np.isin(self.dims['feature'], np.array(data.dims)).all()
        if not dim_samples_exist:
            raise ValueError(f'{self.dims["sample"]} are not present in data array')
        if not dim_features_exist:
            raise ValueError(f'{self.dims["feature"]} are not present in data array')

        # Check if data to be transformed has the same coordinates as the data used to fit the stacker
        if not all([data.coords[dim].equals(self.coords['feature'][dim]) for dim in self.dims['feature']]):  #type: ignore
            raise ValueError('Data to be transformed has different coordinates than the data used to fit.')

        # Stack data and remove NaN features
        data = self._to_2d(data, self.dims['sample'], self.dims['feature'])
        data = data.dropna('feature', how='all')
        data = data.dropna('sample', how='all')
        self.coords_no_nan = {'sample': data.coords['sample'], 'feature': data.coords['feature']}

        # Ensure that no NaNs are present in the data
        if data.isnull().any():
            raise ValueError('Isolated NaNs are present in the data. Please remove them before fitting the model.')

        return data
    
    def inverse_transform_data(self, data: DataArray) -> DataArray:
        ''' Reshape the 2D data (sample x feature) back into its original shape.'''

        data = data.unstack()

        # Reindex data to original coordinates in case that some features at the boundaries were dropped
        data = self._reindex_dim(data, 'feature')
        data = self._reindex_dim(data, 'sample')

        return data
    
    def inverse_transform_components(self, data: DataArray) -> DataArray:
        ''' Reshape the 2D data (mode x feature) back into its original shape.'''

        data = data.unstack()

        # Reindex data to original coordinates in case that some features at the boundaries were dropped
        data = self._reindex_dim(data, 'feature')
        
        return data
    
    def inverse_transform_scores(self, data: DataArray) -> DataArray:
        ''' Reshape the 2D data (sample x mode) back into its original shape.'''

        data = data.unstack()

        # Scores are not to be reindexed since they new data typically has different sample coordinates
        # than the original data used for fitting the model

        return data


class DatasetStacker(DataArrayStacker):
    ''' Converts a Dataset of any dimensionality into a 2D structure.

    This operation generates a reshaped Dataset with two distinct dimensions: 'sample' and 'feature'.
    
    The handling of NaNs is specific: if they are found to populate an entire dimension (be it 'sample' or 'feature'), 
    they are temporarily removed during transformations and subsequently reinstated. 
    However, the presence of isolated NaNs will trigger an error.

    '''

    def _to_2d(self, data: Dataset, sample_dims, feature_dims) -> DataArray:
        ''' Reshape a Dataset to 2D.

        Parameters
        ----------
        data : Dataset
            The data to be reshaped.
        sample_dims : Hashable or Sequence[Hashable]
            The dimensions of the data that will be stacked along the `sample` dimension.
        feature_dims : Hashable or Sequence[Hashable]
            The dimensions of the data that will be stacked along the `feature` dimension.

        Returns
        -------
        data_stacked : DataArray | Dataset
            The reshaped 2d-data.
        '''
        data_da = data.to_stacked_array(new_dim='feature', sample_dims=sample_dims)
        return data_da.stack(sample=sample_dims)
    
    def _reindex_dim(self, data: Dataset, model_dim: str) -> Dataset:
        return super()._reindex_dim(data, model_dim)  # type: ignore

    def fit(self, data: Dataset, sample_dims: Hashable | Sequence[Hashable], feature_dims: Hashable | Sequence[Hashable]):
        return super().fit(data, sample_dims, feature_dims)  # type: ignore

    def transform(self, data: Dataset) -> Dataset:
        return super().transform(data)  # type: ignore

    def inverse_transform_data(self, data: DataArray) -> Dataset:
        ''' Reshape the 2D data (sample x feature) back into its original shape.'''
        data_ds = data.to_unstacked_dataset('variable').unstack()

        # Reindex data to original coordinates in case that some features at the boundaries were dropped
        data_ds = self._reindex_dim(data_ds, 'feature')
        data_ds = self._reindex_dim(data_ds, 'sample')

        return data_ds
    
    def inverse_transform_components(self, data: DataArray) -> Dataset:
        ''' Reshape the 2D data (mode x feature) back into its original shape.'''
        data_ds = data.to_unstacked_dataset('feature').unstack()

        # Reindex data to original coordinates in case that some features at the boundaries were dropped
        data_ds = self._reindex_dim(data_ds, 'feature')

        return data_ds
    
    def inverse_transform_scores(self, data: DataArray) -> DataArray:
        ''' Reshape the 2D data (sample x mode) back into its original shape.'''
        data = data.unstack()

        # Scores are not to be reindexed since they new data typically has different sample coordinates
        # than the original data used for fitting the model

        return data


class DataArrayListStacker():
    ''' Converts a list of DataArrays of any dimensionality into a 2D structure.

    This operation generates a reshaped DataArray with two distinct dimensions: 'sample' and 'feature'.
    
    The handling of NaNs is specific: if they are found to populate an entire dimension (be it 'sample' or 'feature'), 
    they are temporarily removed during transformations and subsequently reinstated. 
    However, the presence of isolated NaNs will trigger an error.
    
    '''
    def __init__(self):
        self.stackers = []

    def fit(
            self,
            data: DataArrayList,
            sample_dims: Hashable | Sequence[Hashable],
            feature_dims: List[Hashable | Sequence[Hashable]]
            ) -> None:
        ''' Fit the stacker to the data.
        
        Parameters
        ----------
        data : DataArray
            The data to be reshaped.
        sample_dims : Hashable or Sequence[Hashable]
            The dimensions of the data that will be stacked along the `sample` dimension.
        feature_dims : Hashable or Sequence[Hashable]
            The dimensions of the data that will be stacked along the `feature` dimension.

        '''        
        # Check input
        if not isinstance(feature_dims, list):
            raise TypeError('feature dims must be a list of the feature dimensions of each DataArray')
        
        sample_dims = ensure_tuple(sample_dims)
        feature_dims = [ensure_tuple(fdims) for fdims in feature_dims]
        self.coords = {
            'sample': [{dim: da.coords[dim] for dim in sample_dims} for da in data],
            'feature': [{dim: da.coords[dim] for dim in fdims} for da, fdims in zip(data, feature_dims)]  #type: ignore
        }

        if len(data) != len(feature_dims):
            err_message = 'Number of data arrays and feature dimensions must be the same. '
            err_message += f'Got {len(data)} data arrays and {len(feature_dims)} feature dimensions'
            raise ValueError(err_message)
        
        for da, fdims in zip(data, feature_dims):
            stacker = DataArrayStacker()
            stacker.fit(da, sample_dims, fdims)
            self.stackers.append(stacker)
    
    def transform(self, data: DataArrayList) -> DataArray:
        ''' Reshape the data into a 2D version.

        Parameters
        ----------
        data: list of DataArrays
            The data to be reshaped.

        Returns
        -------
        DataArray
            The reshaped 2D data.

        '''
        stacked_data_list = []
        idx_coords_size = []
        dummy_feature_coords = []

        # Stack individual DataArrays
        for stacker, da in zip(self.stackers, data):
            stacked_data = stacker.transform(da)
            idx_coords_size.append(stacked_data.coords['feature'].size)
            stacked_data_list.append(stacked_data)
        
        # Create dummy feature coordinates for each DataArray
        idx_range = np.cumsum([0] + idx_coords_size)
        for i in range(len(idx_range) - 1):
            dummy_feature_coords.append(np.arange(idx_range[i], idx_range[i+1]))

        # Replace original feature coordiantes with dummy coordinates
        for i, data in enumerate(stacked_data_list):
            data = data.drop('feature')  # type: ignore
            stacked_data_list[i] = data.assign_coords(feature=dummy_feature_coords[i])  # type: ignore

        self._dummy_feature_coords = dummy_feature_coords
        
        stacked_data = xr.concat(stacked_data_list, dim='feature')
        self.coords_no_nan = {'sample': stacked_data.coords['sample'], 'feature': stacked_data.coords['feature']}
        
        return stacked_data

    def inverse_transform_data(self, data: DataArray) -> DataArrayList:
        dalist = []
        for stacker, features in zip(self.stackers, self._dummy_feature_coords):
            # Select the features corresponding to the current DataArray
            subda = data.sel(feature=features)
            # Replace dummy feature coordinates with original feature coordinates
            subda = subda.assign_coords(feature=stacker.coords_no_nan['feature'])
            subda = subda.set_index(feature=stacker.dims['feature'])
            # NOTE: This is a workaround for the case where the feature dimension is a tuple of length 1
            # the problem is described here: https://github.com/pydata/xarray/discussions/7958
            if len(stacker.dims['feature']) == 1:
                subda = subda.rename(feature=stacker.dims['feature'][0])
            # Inverse transform the data using the corresponding stacker
            subda = stacker.inverse_transform_data(subda)
            dalist.append(subda)
        return dalist
    
    def inverse_transform_components(self, data: DataArray) -> DataArrayList:
        dalist = []
        for stacker, features in zip(self.stackers, self._dummy_feature_coords):
            # Select the features corresponding to the current DataArray
            subda = data.sel(feature=features)
            # Replace dummy feature coordinates with original feature coordinates
            subda = subda.assign_coords(feature=stacker.coords_no_nan['feature'])
            subda = subda.set_index(feature=stacker.dims['feature'])
            # NOTE: This is a workaround for the case where the feature dimension is a tuple of length 1
            # the problem is described here: https://github.com/pydata/xarray/discussions/7958
            if len(stacker.dims['feature']) == 1:
                subda = subda.rename(feature=stacker.dims['feature'][0])
            # Inverse transform the data using the corresponding stacker
            subda = stacker.inverse_transform_components(subda)
            dalist.append(subda)
        return dalist
    
    def inverse_transform_scores(self, data: DataArray) -> DataArray:
        return data.unstack()
            
