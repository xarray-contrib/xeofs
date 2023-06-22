from typing import List, Union, Tuple, Dict, Optional

import numpy as np
import xarray as xr

from ._base_stacker import _BaseStacker

class DataArrayStacker(_BaseStacker):

    def fit(self, da: xr.DataArray, sample_dims: List[str], feature_dims: List[str]):
        self.dims['sample'] = sample_dims
        self.dims['feature'] = feature_dims
        self.coords = {dim: da.coords[dim] for dim in da.dims}

    def stack_data(self, da: xr.DataArray) -> xr.DataArray:
        # Test whether sample and feature dimensions are present in data array
        dim_samples_exist = np.isin(self.dims['sample'], da.dims).all()
        dim_features_exist = np.isin(self.dims['feature'], da.dims).all()
        if not dim_samples_exist:
            raise ValueError(f'{self.dims["sample"]} are not present in data array')
        if not dim_features_exist:
            raise ValueError(f'{self.dims["feature"]} are not present in data array')

        # Stack data and remove NaN features
        da = da.stack(sample=self.dims['sample'], feature=self.dims['feature'])
        da = da.dropna('feature')
        return da
    
    def stack_weights(self, weights: xr.DataArray) -> xr.DataArray:
        # Test whether feature dimensions are present in weights
        dim_features_exist = np.isin(self.dims['feature'], weights.dims).all()
        if not dim_features_exist:
            raise ValueError(f'{self.dims["feature"]} are not present in weights')
        
        return weights.stack(feature=self.dims['feature']).dropna('feature')
    
    def unstack_data(self, da: xr.DataArray) -> xr.DataArray:
        da = da.assign_coords(mode=range(1, da.coords['mode'].size + 1))
        return da.unstack()
    
    def unstack_components(self, da: xr.DataArray) -> xr.DataArray:
        da = da.assign_coords(mode=range(1, da.coords['mode'].size + 1))
        return da.unstack()
    
    def unstack_scores(self, da: xr.DataArray) -> xr.DataArray:
        da = da.assign_coords(mode=range(1, da.coords['mode'].size + 1))
        return da.unstack()


class DataArrayListStacker():
        def __init__(self):
            self.stackers = []

        def fit(self, da_list: List[xr.DataArray], sample_dims: List[str], feature_dims: List[List[str]]) -> None:
            if len(da_list) != len(feature_dims):
                err_message = 'Number of data arrays and feature dimensions must be the same. '
                err_message += f'Got {len(da_list)} data arrays and {len(feature_dims)} feature dimensions'
                raise ValueError(err_message)
            
            for da, fdims in zip(da_list, feature_dims):
                stacker = DataArrayStacker()
                stacker.fit(da, sample_dims, fdims)
                self.stackers.append(stacker)
        
        def stack_data(self, da_list: List[xr.DataArray]) -> List[xr.DataArray]:
            return [stacker.stack_data(da) for stacker, da in zip(self.stackers, da_list)]

        def stack_weights(self, weights_list: List[xr.DataArray]) -> List[xr.DataArray]:
            return [stacker.stack_weights(weights) for stacker, weights in zip(self.stackers, weights_list)]
        
        def unstack_data(self, da_list: List[xr.DataArray]) -> List[xr.DataArray]:
            return [stacker.unstack_data(da) for stacker, da in zip(self.stackers, da_list)]
        
        def unstack_components(self, da_list: List[xr.DataArray]) -> List[xr.DataArray]:
            return [stacker.unstack_components(da) for stacker, da in zip(self.stackers, da_list)]
        
        def unstack_scores(self, da_list: List[xr.DataArray]) -> List[xr.DataArray]:
            return [stacker.unstack_scores(da) for stacker, da in zip(self.stackers, da_list)]
            

class DatasetStacker(_BaseStacker):

    def fit(self, ds: xr.Dataset, sample_dims: List[str], feature_dims: List[str]):
        self.dims['sample'] = sample_dims
        self.dims['feature'] = feature_dims
        self.coords = {dim: ds.coords[dim] for dim in ds.dims}

    def stack_data(self, ds: xr.Dataset) -> xr.Dataset:
        # Test whether sample and feature dimensions are present in dataset
        dim_samples_exist = np.isin(self.dims['sample'], ds.dims).all()
        dim_features_exist = np.isin(self.dims['feature'], ds.dims).all()
        if not dim_samples_exist:
            raise ValueError(f'{self.dims["sample"]} are not present in data array')
        if not dim_features_exist:
            raise ValueError(f'{self.dims["feature"]} are not present in data array')

        # Stack data and remove NaN features
        ds = ds.to_stacked_array(new_dim='feature', sample_dims=self.dims['sample'])
        ds = ds.stack(sample=self.dims['sample'])
        ds = ds.dropna('feature')
        
        return ds
        
    def stack_weights(self, weights: xr.Dataset) -> xr.Dataset:
        # Test whether feature dimensions are present in weights
        dim_features_exist = np.isin(self.dims['feature'], weights.dims).all()
        if not dim_features_exist:
            raise ValueError(f'{self.dims["feature"]} are not present in weights')
        
        weights = weights.expand_dims({'sample': [0]})
        weights = weights.to_stacked_array(new_dim='feature', sample_dims=('sample',)).squeeze()
        return weights.dropna('feature')
    
    def unstack_data(self, ds: xr.Dataset) -> xr.Dataset:
        ds = ds.unstack('sample')
        ds = ds.to_unstacked_dataset('feature', 'variable')
        ds = ds.unstack('feature')
        ds = ds.assign_coords(mode=range(1, ds.coords['mode'].size + 1))
        return ds
    
    def unstack_components(self, ds: xr.Dataset) -> xr.Dataset:
        ds = ds.to_unstacked_dataset('feature').unstack()
        ds = ds.assign_coords(mode=range(1, ds.coords['mode'].size + 1))
        return ds
    
    def unstack_scores(self, ds: xr.Dataset) -> xr.Dataset:
        ds = ds.unstack()
        ds = ds.assign_coords(mode=range(1, ds.coords['mode'].size + 1))
        return ds
