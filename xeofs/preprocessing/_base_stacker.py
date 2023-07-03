from abc import ABC, abstractmethod

from typing import Sequence, Hashable


class _BaseStacker(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(
            self,
            data,
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
        raise NotImplementedError

    @abstractmethod
    def transform(self, data):
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
        raise NotImplementedError

    @abstractmethod
    def inverse_transform_data(self, data):
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
    def inverse_transform_components(self, data):
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
    
    def inverse_transform_scores(self, data):
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
        