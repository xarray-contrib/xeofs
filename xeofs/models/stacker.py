from typing import List, Union, Tuple, Dict, Optional, Sequence, Hashable

import numpy as np
import xarray as xr

from ._base_stacker import _BaseStacker
from ..utils.data_types import ModelDims, DataArray, DataArrayList, Dataset
from ..utils.sanity_checks import ensure_tuple

class DataArrayStacker(_BaseStacker):
    ''' Reshape any N-dimensional DataArray into a 2D version.
    
    The new object has two dimensions, `sample` and `feature`.

    '''

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
        self.dims: ModelDims = {'sample': sample_dims, 'feature': feature_dims}
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
            
        '''
        # Test whether sample and feature dimensions are present in data array
        dim_samples_exist = np.isin(self.dims['sample'], np.array(data.dims)).all()
        dim_features_exist = np.isin(self.dims['feature'], np.array(data.dims)).all()
        if not dim_samples_exist:
            raise ValueError(f'{self.dims["sample"]} are not present in data array')
        if not dim_features_exist:
            raise ValueError(f'{self.dims["feature"]} are not present in data array')

        # Stack data and remove NaN features
        data = data.stack(sample=self.dims['sample'], feature=self.dims['feature'])
        data = data.dropna('feature')
        self.coords_no_nan = {'sample': data.coords['sample'], 'feature': data.coords['feature']}

        return data
    
    def inverse_transform_data(self, data: DataArray) -> DataArray:
        ''' Reshape the 2D data (sample x feature) back into its original shape.
        

        '''
        data = data.unstack()

        # Reindex data to original coordinates in case that some features at the boundaries were dropped
        # check if coordinates in self.coords['feature'] have different length from data.coords['feature']
        # if so, reindex data.coords['feature'] to self.coords['feature']
        for dim in self.coords['feature'].keys():
            if self.coords['feature'][dim].size != data.coords[dim].size:
                data = data.reindex({dim: self.coords['feature'][dim]})


        return data
    
    def inverse_transform_components(self, data: DataArray) -> DataArray:
        ''' Reshape the 2D data (mode x feature) back into its original shape.
        
        Parameters
        ----------
        data : DataArray
            The data to be reshaped.

        Returns
        -------
        DataArray
            The reshaped data.
            
        '''

        data = data.unstack()

        # Reindex data to original coordinates in case that some features at the boundaries were dropped
        # check if coordinates in self.coords['feature'] have different length from data.coords['feature']
        # if so, reindex data.coords['feature'] to self.coords['feature']
        for dim in self.coords['feature'].keys():
            if self.coords['feature'][dim].size != data.coords[dim].size:
                data = data.reindex({dim: self.coords['feature'][dim]})

        
        return data
    
    def inverse_transform_scores(self, data: DataArray) -> DataArray:
        ''' Reshape the 2D data (sample x mode) back into its original shape.

        Parameters
        ----------
        data : DataArray
            The data to be reshaped.

        Returns
        -------
        DataArray
            The reshaped data.
            
        '''
        return data.unstack()


class DataArrayListStacker():
    ''' Reshape a list of N-dimensional DataArrays into a 2D version.
    
    The new object has two dimensions, `sample` and `feature`.
    
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
            'feature': [{dim: da.coords[dim] for dim in fdims} for da, fdims in zip(data, feature_dims)]
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

        return xr.concat(stacked_data_list, dim='feature')

    def inverse_transform_data(self, data: DataArray) -> DataArrayList:
        dalist = []
        for stacker, features in zip(self.stackers, self._dummy_feature_coords):
            subda = data.sel(feature=features)
            subda = subda.assign_coords(feature=stacker.coords_no_nan['feature'])
            subda = subda.set_index(feature=stacker.dims['feature'])
            subda = stacker.inverse_transform_data(subda)
            dalist.append(subda)
        return dalist
    
    def inverse_transform_components(self, data: DataArray) -> DataArrayList:
        dalist = []
        for stacker, features in zip(self.stackers, self._dummy_feature_coords):
            subda = data.sel(feature=features)
            subda = subda.assign_coords(feature=stacker.coords_no_nan['feature'])
            subda = subda.set_index(feature=stacker.dims['feature'])
            subda = stacker.inverse_transform_components(subda)
            dalist.append(subda)
        return dalist
    
    def inverse_transform_scores(self, data: DataArray) -> DataArray:
        return data.unstack()
            

class DatasetStacker(_BaseStacker):
    ''' Reshape any N-dimensional Dataset into a 2D version.
    
    The new object has two dimensions, `sample` and `feature`.

    '''
    def fit(
            self,
            data: Dataset,
            sample_dims: Hashable | Sequence[Hashable],
            feature_dims: Hashable | Sequence[Hashable]
            ):

        sample_dims = ensure_tuple(sample_dims)
        feature_dims = ensure_tuple(feature_dims)
        self.dims: ModelDims = {'sample': sample_dims, 'feature': feature_dims}
        self.coords = {
            'sample': {dim: data.coords[dim] for dim in sample_dims},
            'feature': {dim: data.coords[dim] for dim in feature_dims}
        }
        

    def transform(self, data: Dataset) -> DataArray:
        # Test whether sample and feature dimensions are present in dataset
        dim_samples_exist = np.isin(self.dims['sample'], np.array(data.dims)).all()
        dim_features_exist = np.isin(self.dims['feature'], np.array(data.dims)).all()
        if not dim_samples_exist:
            raise ValueError(f'{self.dims["sample"]} are not present in data array')
        if not dim_features_exist:
            raise ValueError(f'{self.dims["feature"]} are not present in data array')

        # Stack data and remove NaN features
        data_da = data.to_stacked_array(new_dim='feature', sample_dims=self.dims['sample'])
        data_da = data_da.stack(sample=self.dims['sample'])
        data_da = data_da.dropna('feature')
        self.coords_no_nan = {'sample': data_da.coords['sample'], 'feature': data_da.coords['feature']}
        
        return data_da

    def inverse_transform_data(self, data: DataArray) -> Dataset:
        data = data.unstack('sample')
        data_ds = data.to_unstacked_dataset('feature', 'variable')
        data_ds = data_ds.unstack('feature')

        # Reindex data to original coordinates in case that some features at the boundaries were dropped
        # check if coordinates in self.coords['feature'] have different length from data.coords['feature']
        # if so, reindex data.coords['feature'] to self.coords['feature']
        for dim in self.coords['feature'].keys():
            if self.coords['feature'][dim].size != data.coords[dim].size:
                data_ds = data_ds.reindex({dim: self.coords['feature'][dim]})

        return data_ds
    
    def inverse_transform_components(self, data: DataArray) -> Dataset:
        data_ds = data.to_unstacked_dataset('feature').unstack()

        # Reindex data to original coordinates in case that some features at the boundaries were dropped
        # check if coordinates in self.coords['feature'] have different length from data.coords['feature']
        # if so, reindex data.coords['feature'] to self.coords['feature']
        for dim in self.coords['feature'].keys():
            if self.coords['feature'][dim].size != data.coords[dim].size:
                data_ds = data_ds.reindex({dim: self.coords['feature'][dim]})

        return data_ds
    
    def inverse_transform_scores(self, data: DataArray) -> DataArray:
        data = data.unstack()
        return data

