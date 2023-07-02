from abc import ABC, abstractmethod

import numpy as np
import scipy as sc
import xarray as xr
from typing import Optional, Union, List, Tuple

from ..utils.rotation import promax


class _BaseRotator():
    '''Rotates a solution obtained from an EOF model.

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

    def __init__(self, n_rot: int, power: int = 1, max_iter: int = 1000, rtol: float = 1e-8):
        self._params = {
            'n_rot': n_rot,
            'power': power,
            'max_iter': max_iter,
            'rtol': rtol,
        }
    
    @abstractmethod
    def fit(self, model):
        '''Fit the model.'''

        # VARIABLES TO BE DEFINED
        # self._model
        # self._rotation_matrix --> rotation matrix to be applied to the loadings
        # self._idx_expvar --> order of explained variance so that modes can be reordered
        # according to highest explained variance

        # OTHER VARIABLES THAT DEPEND ON THE SPECIFIC MODEL TO BE ROTATED
        # self._explained_variance
        # self._explained_variance_ratio | self._squared_covariance_fraction
        # self._components
        # self._scores
        raise NotImplementedError
    
    @abstractmethod
    def transform(self, data):
        '''Transform the model.'''
        raise NotImplementedError
    
    @abstractmethod
    def inverse_transform(self, mode: int | List[int] | slice = slice(None)):
        '''Inverse transform the model.'''
        raise NotImplementedError
    
    
    def _get_rotation_matrix(self, inverse_transpose : bool = False) -> xr.DataArray:
        '''Return the rotation matrix.

        Parameters
        ----------
        inverse_transpose : bool, default=False
            Determines whether to return the rotation matrix or its inverse transposed. 
            For orthogonal rotations (e.g., Varimax), the inverse transpose is equivalent 
            to the rotation matrix itself. However, for oblique rotations (e.g., Promax), it may differ.

        Returns
        -------
        rotation_matrix : xr.DataArray

        '''
        R = self._rotation_matrix  # type: ignore
        if inverse_transpose and self._params['power'] > 1:
            # inverse matrix
            R = xr.apply_ufunc(
                np.linalg.pinv,
                R,
                input_core_dims=[['mode','mode1']],
                output_core_dims=[['mode','mode1']]
            )
            # transpose matrix
            R = R.conj().T
        return R  # type: ignore

