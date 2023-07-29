from typing import List, Sequence, Hashable

import numpy as np
import pandas as pd
import xarray as xr

from xeofs.utils.data_types import DataArray

from ._base_stacker import _BaseStacker
from ..utils.data_types import DataArray, DataArrayList, Dataset, SingleDataObject, AnyDataObject
from ..utils.sanity_checks import ensure_tuple


class SingleDataStacker(_BaseStacker):

    def __init__(self):
        super().__init__()
    
    def _verify_dims(self, data: SingleDataObject):
        ''' Verify that the dimensions of the data are consistent with the dimensions used to fit the stacker.'''
        # Test whether sample and feature dimensions are present in data array
        if not (set(self.dims_out_['sample'] + self.dims_out_['feature']) == set(data.dims)):
            raise ValueError(f'One or more dimensions in {self.dims_out_["sample"] + self.dims_out_["feature"]} are not present in data.')

    def _stack(self, data: SingleDataObject, sample_dims, feature_dims) -> DataArray:
        ''' Reshape a SingleDataObject to 2D DataArray.'''
        raise NotImplementedError
    
    def _unstack(self, data: SingleDataObject) -> SingleDataObject:
        ''' Unstack `sample` and `feature` dimension of an DataArray to its original dimensions.

        Parameters
        ----------
        data : DataArray
            The data to be unstacked.

        Returns
        -------
        data_unstacked : DataArray
            The unstacked data.
        '''
        # pass if feature/sample dimensions do not exist in data
        if 'feature' in data.dims:
            # If sample dimensions is one dimensional, rename is sufficient, otherwise unstack
            if len(self.dims_out_['feature']) == 1:
                data = data.rename({'feature': self.dims_out_['feature'][0]})
            else:
                data = data.unstack('feature')
        
        if 'sample' in data.dims:
            # If sample dimensions is one dimensional, rename is sufficient, otherwise unstack
            if len(self.dims_out_['sample']) == 1:
                data = data.rename({'sample': self.dims_out_['sample'][0]})
            else:
                data = data.unstack('sample')
        
        # Reorder dimensions to original order; catch ('mode') dimensions via ellipsis
        order_input_dims = [valid_dim for valid_dim in self.dims_in_ if valid_dim in data.dims]
        data = data.transpose(..., *order_input_dims)
        
        return data

    def _reindex_dim(self, data: SingleDataObject, stacked_dim: str) -> SingleDataObject:
        ''' Reindex data to original coordinates in case that some features at the boundaries were dropped
        
        Parameters
        ----------
        data : DataArray
            The data to be reindex.
        stacked_dim : str ['sample', 'feature']
            The dimension to be reindexed.
            
        Returns
        -------
        DataArray
            The reindexed data.
            
        '''
        # check if coordinates in self.coords have different length from data.coords
        # if so, reindex data.coords to self.coords
        # input_dim : dimensions of input data
        # stacked_dim : dimensions of model data i.e. sample or feature
        dims_in = self.dims_out_[stacked_dim]
        for dim in dims_in:
            if self.coords_in_[dim].size != data.coords[dim].size:
                data = data.reindex({dim: self.coords_in_[dim]}, copy=False)

        return data


    def fit_transform(self, data: SingleDataObject, sample_dims: Hashable | Sequence[Hashable], feature_dims: Hashable | Sequence[Hashable]) -> DataArray:
        ''' Fit the stacker and transform data to 2D.
        
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
        DataArray
            The reshaped data.

        Raises
        ------
        ValueError
            If any of the dimensions in `sample_dims` or `feature_dims` are not present in the data.
        ValueError
            If data to be transformed has individual NaNs.
        ValueError
            If data is empty
        
        '''

        sample_dims = ensure_tuple(sample_dims)
        feature_dims = ensure_tuple(feature_dims)
    
        # Test whether any of the dimensions in sample_dims or feature_dims is missing in the data
        if not (set(sample_dims + feature_dims) == set(data.dims)):
            raise ValueError(f'One or more dimensions in {sample_dims + feature_dims} are not present in data dimensions: {data.dims}')

        # Set in/out dimensions
        self.dims_in_ = data.dims
        self.dims_out_ = {'sample': sample_dims, 'feature': feature_dims}

        # Set in/out coordinates
        self.coords_in_ = {dim: data.coords[dim] for dim in data.dims}
        # Stack data and remove NaN features
        da: DataArray = self._stack(data, self.dims_out_['sample'], self.dims_out_['feature'])
        da = da.dropna('feature', how='all')
        da = da.dropna('sample', how='all')
        self.coords_out_ = {'sample': da.coords['sample'], 'feature': da.coords['feature']}

        # Ensure that no NaNs are present in the data
        if da.isnull().any():
            raise ValueError('Isolated NaNs are present in the data. Please remove them before fitting the model.')

        # Ensure that data is not empty
        if da.size == 0:
            raise ValueError('Data is empty.')

        return da

    def transform(self, data: SingleDataObject) -> DataArray:
        ''' Transform new "unseen" data to 2D version.
        
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
        ValueError
            If data is empty
            
        '''
        # Test whether sample and feature dimensions are present in data array
        if not (set(self.dims_out_['sample'] + self.dims_out_['feature']) == set(data.dims)):
            raise ValueError(f'One or more dimensions in {self.dims_out_["sample"] + self.dims_out_["feature"]} are not present in data.')

        # Check if data to be transformed has the same feature coordinates as the data used to fit the stacker
        if not all([data.coords[dim].equals(self.coords_in_[dim]) for dim in self.dims_out_['feature']]):  #type: ignore
            raise ValueError('Data to be transformed has different coordinates than the data used to fit.')

        # Stack data and remove NaN features
        da: DataArray = self._stack(data, self.dims_out_['sample'], self.dims_out_['feature'])
        da = da.dropna('feature', how='all')
        da = da.dropna('sample', how='all')

        # Ensure that no NaNs are present in the data
        if da.isnull().any():
            raise ValueError('Isolated NaNs are present in the data. Please remove them before fitting the model.')

        # Ensure that data is not empty
        if da.size == 0:
            raise ValueError('Data is empty.')

        return da


class SingleDataArrayStacker(SingleDataStacker):
    ''' Converts a DataArray of any dimensionality into a 2D structure.

    This operation generates a reshaped DataArray with two distinct dimensions: 'sample' and 'feature'.
    
    The handling of NaNs is specific: if they are found to populate an entire dimension (be it 'sample' or 'feature'), 
    they are temporarily removed during transformations and subsequently reinstated. 
    However, the presence of isolated NaNs will trigger an error.
        
    '''

    @staticmethod
    def _stack(data: DataArray, sample_dims, feature_dims) -> DataArray:
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
        # Raise error if dimensions feature/sample already exists in data
        if 'feature' in data.dims:
            raise ValueError('The dimension name `feature` conflicts with internal usage. Please rename this dimension before the analysis.')
        if 'sample' in data.dims:
            raise ValueError('The dimension name `sample` conflicts with internal usage. Please rename this dimension before the analysis.')

        # Rename if sample/feature dimensions is one dimensional, otherwise stack
        if len(feature_dims) == 1:
            data = data.rename({feature_dims[0]: 'feature'})
        else:
            data = data.stack(feature=feature_dims)

        if len(sample_dims) == 1:
            data = data.rename({sample_dims[0]: 'sample'})
        else:
            data = data.stack(sample=sample_dims)

        return data.transpose('sample', 'feature')

    def _unstack(self, data: DataArray) -> DataArray:
        ''' Unstack `sample` and `feature` dimension of an DataArray to its original dimensions.

        Parameters
        ----------
        data : DataArray
            The data to be unstacked.

        Returns
        -------
        data_unstacked : DataArray
            The unstacked data.
        '''
        return super()._unstack(data)

    def _reindex_dim(self, data: DataArray,  stacked_dim : str) -> DataArray:
        return super()._reindex_dim(data, stacked_dim)

    def fit_transform(
            self,
            data: DataArray,
            sample_dims: Hashable | Sequence[Hashable],
            feature_dims: Hashable | Sequence[Hashable]
            ) -> DataArray:
        return super().fit_transform(data, sample_dims, feature_dims)

    def transform(self, data: DataArray) -> DataArray:
        return super().transform(data)

    
    def inverse_transform_data(self, data: DataArray) -> DataArray:
        ''' Reshape the 2D data (sample x feature) back into its original shape.'''

        data = self._unstack(data)

        # Reindex data to original coordinates in case that some features at the boundaries were dropped
        data = self._reindex_dim(data, 'feature')
        data = self._reindex_dim(data, 'sample')

        return data
    
    def inverse_transform_components(self, data: DataArray) -> DataArray:
        ''' Reshape the 2D data (mode x feature) back into its original shape.'''

        data = self._unstack(data)

        # Reindex data to original coordinates in case that some features at the boundaries were dropped
        data = self._reindex_dim(data, 'feature')
        
        return data
    
    def inverse_transform_scores(self, data: DataArray) -> DataArray:
        ''' Reshape the 2D data (sample x mode) back into its original shape.'''

        data = self._unstack(data)

        # Scores are not to be reindexed since they new data typically has different sample coordinates
        # than the original data used for fitting the model

        return data


class SingleDatasetStacker(SingleDataStacker):
    ''' Converts a Dataset of any dimensionality into a 2D structure.

    This operation generates a reshaped Dataset with two distinct dimensions: 'sample' and 'feature'.
    
    The handling of NaNs is specific: if they are found to populate an entire dimension (be it 'sample' or 'feature'), 
    they are temporarily removed during transformations and subsequently reinstated. 
    However, the presence of isolated NaNs will trigger an error.

    '''

    def _stack(self, data: Dataset, sample_dims, feature_dims) -> DataArray:
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
        # Raise error if dimensions feature/sample already exists in data
        if 'feature' in data.dims:
            raise ValueError('The dimension name `feature` conflicts with internal usage. Please rename this dimension before the analysis.')
        if 'sample' in data.dims:
            raise ValueError('The dimension name `sample` conflicts with internal usage. Please rename this dimension before the analysis.')

        # Convert Dataset -> DataArray, stacking all non-sample dimensions to feature dimension, including data variables
        data_da: DataArray = data.to_stacked_array(new_dim='feature', sample_dims=sample_dims)

        # Rename if sample dimensions is one dimensional, otherwise stack
        if len(sample_dims) == 1:
            data_da = data_da.rename({sample_dims[0]: 'sample'})
        else:
            data_da = data_da.stack(sample=sample_dims)

        return data_da.transpose('sample', 'feature')

    def _unstack(self, data: SingleDataObject) -> SingleDataObject:
        ''' Unstack `sample` and `feature` dimension of an DataArray to its original dimensions.'''
        return super()._unstack(data)

    def _reindex_dim(self, data: Dataset, model_dim: str) -> Dataset:
        return super()._reindex_dim(data, model_dim)

    def fit_transform(self, data: Dataset, sample_dims: Hashable | Sequence[Hashable], feature_dims: Hashable | Sequence[Hashable] | List[Sequence[Hashable]]) -> xr.DataArray:
        return super().fit_transform(data, sample_dims, feature_dims)

    def transform(self, data: Dataset) -> DataArray:
        return super().transform(data)

    def inverse_transform_data(self, data: DataArray) -> Dataset:
        ''' Reshape the 2D data (sample x feature) back into its original shape.'''
        data_ds: Dataset = data.to_unstacked_dataset('variable')
        
        data_ds = self._unstack(data_ds)

        # Reindex data to original coordinates in case that some features at the boundaries were dropped
        data_ds = self._reindex_dim(data_ds, 'feature')
        data_ds = self._reindex_dim(data_ds, 'sample')

        return data_ds
    
    def inverse_transform_components(self, data: DataArray) -> Dataset:
        ''' Reshape the 2D data (mode x feature) back into its original shape.'''
        data_ds = data.to_unstacked_dataset('feature')
        
        data_ds = self._unstack(data_ds)

        # Reindex data to original coordinates in case that some features at the boundaries were dropped
        data_ds = self._reindex_dim(data_ds, 'feature') 

        return data_ds
    
    def inverse_transform_scores(self, data: DataArray) -> DataArray:
        ''' Reshape the 2D data (sample x mode) back into its original shape.'''
        data = self._unstack(data)

        # Scores are not to be reindexed since they new data typically has different sample coordinates
        # than the original data used for fitting the model

        return data


class ListDataArrayStacker(_BaseStacker):
    ''' Converts a list of DataArrays of any dimensionality into a 2D structure.

    This operation generates a reshaped DataArray with two distinct dimensions: 'sample' and 'feature'.
    
    The handling of NaNs is specific: if they are found to populate an entire dimension (be it 'sample' or 'feature'), 
    they are temporarily removed during transformations and subsequently reinstated. 
    However, the presence of isolated NaNs will trigger an error.

    At a minimum, the `sample` dimension must be present in all DataArrays. The `feature` dimension can be different
    for each DataArray and must be specified as a list of dimensions.
    
    '''
    def __init__(self):
        self.stackers = []

    def fit_transform(
            self,
            data: DataArrayList,
            sample_dims: Hashable | Sequence[Hashable],
            feature_dims: Hashable | Sequence[Hashable] | List[Sequence[Hashable]]
            ) -> DataArray:
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

        if len(data) != len(feature_dims):
            err_message = 'Number of data arrays and feature dimensions must be the same. '
            err_message += f'Got {len(data)} data arrays and {len(feature_dims)} feature dimensions'
            raise ValueError(err_message)
        
        # Set in/out dimensions
        self.dims_in_ = [da.dims for da in data]
        self.dims_out_ = {'sample': sample_dims, 'feature': feature_dims}

        # Set in/out coordinates
        self.coords_in_ = [{dim: coords for dim, coords in da.coords.items()} for da in data]

        for da, fdims in zip(data, feature_dims):
            stacker = SingleDataArrayStacker()
            da_stacked = stacker.fit_transform(da, sample_dims, fdims)
            self.stackers.append(stacker)

        stacked_data_list = []
        idx_coords_size = []
        dummy_feature_coords = []

        # Stack individual DataArrays
        for da, fdims in zip(data, feature_dims):
            stacker = SingleDataArrayStacker()
            da_stacked = stacker.fit_transform(da, sample_dims, fdims)
            idx_coords_size.append(da_stacked.coords['feature'].size)
            stacked_data_list.append(da_stacked)
        
        # Create dummy feature coordinates for each DataArray
        idx_range = np.cumsum([0] + idx_coords_size)
        for i in range(len(idx_range) - 1):
            dummy_feature_coords.append(np.arange(idx_range[i], idx_range[i+1]))

        # Replace original feature coordiantes with dummy coordinates
        for i, data in enumerate(stacked_data_list):
            data = data.drop('feature')  # type: ignore
            stacked_data_list[i] = data.assign_coords(feature=dummy_feature_coords[i])  # type: ignore

        self._dummy_feature_coords = dummy_feature_coords

        stacked_data_list = xr.concat(stacked_data_list, dim='feature')

        self.coords_out_ = {
            'sample': stacked_data_list.coords['sample'], 
            'feature': stacked_data_list.coords['feature']
        }
        return stacked_data_list

    
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

        # Stack individual DataArrays
        for i, (stacker, da) in enumerate(zip(self.stackers, data)):
            stacked_data = stacker.transform(da)
            stacked_data = stacked_data.drop('feature') 
            # Replace original feature coordiantes with dummy coordinates
            stacked_data.coords.update({'feature': self._dummy_feature_coords[i]})            
            stacked_data_list.append(stacked_data)
        
        return xr.concat(stacked_data_list, dim='feature')

    def inverse_transform_data(self, data: DataArray) -> DataArrayList:
        ''' Reshape the 2D data (sample x feature) back into its original shape.'''
        dalist = []
        for stacker, features in zip(self.stackers, self._dummy_feature_coords):
            # Select the features corresponding to the current DataArray
            subda = data.sel(feature=features)
            # Replace dummy feature coordinates with original feature coordinates
            subda = subda.assign_coords(feature=stacker.coords_out_['feature'])
            
            # In case of MultiIndex we have to set the index to the feature dimension again
            if isinstance(subda.indexes['feature'], pd.MultiIndex):
                subda = subda.set_index(feature=stacker.dims_out_['feature'])
            else:
                # NOTE: This is a workaround for the case where the feature dimension is a tuple of length 1
                # the problem is described here: https://github.com/pydata/xarray/discussions/7958
                subda = subda.rename(feature=stacker.dims_out_['feature'][0])

            # Inverse transform the data using the corresponding stacker
            subda = stacker.inverse_transform_data(subda)
            dalist.append(subda)
        return dalist
    
    def inverse_transform_components(self, data: DataArray) -> DataArrayList:
        ''' Reshape the 2D data (mode x feature) back into its original shape.'''
        dalist = []
        for stacker, features in zip(self.stackers, self._dummy_feature_coords):
            # Select the features corresponding to the current DataArray
            subda = data.sel(feature=features)
            # Replace dummy feature coordinates with original feature coordinates
            subda = subda.assign_coords(feature=stacker.coords_out_['feature'])

            # In case of MultiIndex we have to set the index to the feature dimension again
            if isinstance(subda.indexes['feature'], pd.MultiIndex):
                subda = subda.set_index(feature=stacker.dims_out_['feature'])
            else:
                # NOTE: This is a workaround for the case where the feature dimension is a tuple of length 1
                # the problem is described here: https://github.com/pydata/xarray/discussions/7958
                subda = subda.rename(feature=stacker.dims_out_['feature'][0])

            # Inverse transform the data using the corresponding stacker
            subda = stacker.inverse_transform_components(subda)
            dalist.append(subda)
        return dalist
    
    def inverse_transform_scores(self, data: DataArray) -> DataArray:
        ''' Reshape the 2D data (sample x mode) back into its original shape.'''
        return self.stackers[0].inverse_transform_scores(data)
        
            
