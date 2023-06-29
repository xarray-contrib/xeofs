import numpy as np
import xarray as xr

from ._base_model import _BaseModel
from .decomposer import Decomposer
from ..utils.data_types import XarrayData, DataArrayList
from ..utils.xarray_utils import total_variance


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

        self._total_variance = total_variance(self.data)

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
