from abc import ABC, abstractmethod
from typing import Sequence, Hashable, List

import xarray as xr


class _BaseStacker(ABC):
    ''' Abstract base class for stacking data into a 2D array.

    Every multi-dimensional array is be reshaped into a 2D array with the
    dimensions (sample x feature).

    
    Attributes
    ----------
    dims_in_ : tuple
        The dimensions of the data used to fit the stacker.
    dims_out_ : dict['sample': ..., 'feature': ...]
        The dimensions of the stacked data.
    coords_in_ : dict
        The coordinates of the data used to fit the stacker.
    coords_out_ : dict['sample': ..., 'feature': ...]
        The coordinates of the stacked data. Typically consist of MultiIndex.

    '''
    def __init__(self):
        pass

    def fit(self, data, sample_dims: Hashable | Sequence[Hashable], feature_dims: Hashable | Sequence[Hashable] | List[Sequence[Hashable]]):
        ''' Invoking a `fit` operation for a stacker object isn't practical because it requires stacking the data, 
        only to ascertain the output dimensions. This step is computationally expensive and unnecessary. 
        Therefore, instead of using a separate `fit` method, we combine the fit and transform steps 
        into the `fit_transform` method for efficiency. However, to maintain consistency with other classes 
        that do utilize a `fit` method, we retain the `fit` method here, albeit unimplemented.

        '''
        raise NotImplementedError('Stacker does not implement fit method. Use fit_transform instead.')
        

    @abstractmethod
    def fit_transform(
            self,
            data,
            sample_dims: Hashable | Sequence[Hashable],
            feature_dims: Hashable | Sequence[Hashable] | List[Sequence[Hashable]]
            ) -> xr.DataArray:
        ''' Fit the stacker to the data and then transform the data.
        
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
        '''
        raise NotImplementedError

    @abstractmethod
    def transform(self, data) -> xr.DataArray:
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
            If the data to be transformed has different feature coordinates than the data used to fit the stacker.
        ValueError
            If the data to be transformed has individual NaNs.
            
        '''
        raise NotImplementedError

    @abstractmethod
    def inverse_transform_data(self, data: xr.DataArray):
        ''' Reshape the 2D data (sample x feature) back into its original shape.
        
        Parameters
        ----------
        data : DataArray
            The data to be reshaped.
            
        Returns
        -------
        DataArray
            The reshaped data.
            
        '''
        raise NotImplementedError
    
    @abstractmethod
    def inverse_transform_components(self, data: xr.DataArray):
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
        raise NotImplementedError
    
    @abstractmethod
    def inverse_transform_scores(self, data: xr.DataArray):
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
        raise NotImplementedError
        