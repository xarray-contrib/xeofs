import numpy as np
import xarray as xr
from typing import Optional, Union, List, Tuple

from .eof import EOF
from .complex_eof import ComplexEOF
from ._base_rotator import _BaseRotator
from ..utils.rotation import promax
from ..utils.data_types import XarrayData, DataArrayList, Dataset, DataArray

class EOFRotator(_BaseRotator):
    '''Rotates a solution obtained from ``xe.models.EOF``.

    Parameters
    ----------
    n_rot : int
        Number of modes to be rotated.
    power : int
        Defines the power of Promax rotation. Choosing ``power=1`` equals
        a Varimax solution (the default is 1).
    max_iter : int
        Number of maximal iterations for obtaining the rotation matrix
        (the default is 1000).
    rtol : float
        Relative tolerance to be achieved for early stopping the iteration
        process (the default is 1e-8).

    '''

    def transform(self, data: XarrayData | DataArrayList) -> XarrayData | DataArrayList:

        n_rot = self._params['n_rot']
        svals = self._model._singular_values.sel(mode=slice(1, self._params['n_rot']))
        components = self._model._components.sel(mode=slice(1, n_rot))

        # Preprocess the data
        data = self._model.scaler.transform(data)  #type: ignore
        data = self._model.stacker.transform(data)  #type: ignore

        # Compute non-rotated scores by project the data onto non-rotated components
        projections = xr.dot(data, components) / svals
        projections.name = 'scores'

        # Rotate the scores
        R = self._get_rotation_matrix(inverse_transpose=True)
        projections = xr.dot(projections, R, dims='mode1')
        # Reorder according to variance
        projections = projections.isel(mode=self._idx_expvar).assign_coords(mode=projections.mode)

        # Unstack the projections
        projections = self._model.stacker.inverse_transform_scores(projections)
        return projections      
 
class ComplexEOFRotator(_BaseRotator):
    '''Rotates a solution obtained from ``xe.models.ComplexEOF``.

    Parameters
    ----------
    n_rot : int
        Number of modes to be rotated.
    power : int
        Defines the power of Promax rotation. Choosing ``power=1`` equals
        a Varimax solution (the default is 1).
    max_iter : int
        Number of maximal iterations for obtaining the rotation matrix
        (the default is 1000).
    rtol : float
        Relative tolerance to be achieved for early stopping the iteration
        process (the default is 1e-8).

    '''

    def components_amplitude(self) -> DataArray | Dataset | DataArrayList:
        '''Compute the amplitude of the components.

        Returns
        -------
        xr.DataArray
            Amplitude of the components.

        '''
        comps = abs(self._components)
        comps.name = 'amplitude'
        comps = self._model.stacker.inverse_transform_components(comps)
        return comps

    def components_phase(self) -> DataArray | Dataset | DataArrayList:
        '''Compute the phase of the components.

        Returns
        -------
        xr.DataArray
            Phase of the components.

        '''
        comps = np.angle(self._components)
        comps.name = 'phase'
        comps = self._model.stacker.inverse_transform_components(comps)
        return comps
    
    def scores_amplitude(self) -> DataArray | Dataset | DataArrayList:
        '''Compute the amplitude of the scores.

        Returns
        -------
        xr.DataArray
            Amplitude of the scores.

        '''
        scores = abs(self._scores)
        scores.name = 'amplitude'
        scores = self._model.stacker.inverse_transform_scores(scores)
        return scores
    
    def scores_phase(self) -> DataArray | Dataset | DataArrayList:
        '''Compute the phase of the scores.

        Returns
        -------
        xr.DataArray
            Phase of the scores.

        '''
        scores = np.angle(self._scores)
        scores.name = 'phase'
        scores = self._model.stacker.inverse_transform_scores(scores)
        return scores



class RotatorFactory:
    '''Factory class for creating rotators.

    Parameters
    ----------
    n_rot : int
        Number of modes to be rotated.
    power : int
        Defines the power of Promax rotation. Choosing ``power=1`` equals
        a Varimax solution (the default is 1).
    max_iter : int
        Number of maximal iterations for obtaining the rotation matrix
        (the default is 1000).
    rtol : float
        Relative tolerance to be achieved for early stopping the iteration
        process (the default is 1e-8).
    
    '''
    def __init__(self, **kwargs):
        self.params = kwargs
        self._valid_types = (EOF, ComplexEOF)


    def create_rotator(self, model: EOF | ComplexEOF) -> EOFRotator | ComplexEOFRotator:
        '''Create a rotator for the given model.
        
        Parameters
        ----------
        model : xeofs model
            Model to be rotated.
        
        Returns
        -------
        xeofs Rotator
            Rotator for the given model.
        '''
        # We need to check the type of the model instead of isinstance because
        # of inheritance.
        if type(model) == EOF:
            return EOFRotator(**self.params)
        elif type(model) == ComplexEOF:
            return ComplexEOFRotator(**self.params)
        else:
            err_msg = f'Invalid model type. Valid types are {self._valid_types}.'
            raise TypeError(err_msg)
