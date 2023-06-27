from abc import ABC, abstractmethod

import numpy as np
import scipy as sc
import xarray as xr
from typing import Optional, Union, List, Tuple

from ._base_model import EOF
from ..utils.rotation import promax
from ..utils.tools import get_mode_selector


class _BaseRotator(ABC):
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

    def __init__(self, n_rot: int, power: int = 1, max_iter: int = 1000, rtol: float = 1e-8):
        self._params = {
            'n_rot': n_rot,
            'power': power,
            'max_iter': max_iter,
            'rtol': rtol,
        }
    
    @abstractmethod
    def fit(self, model: EOF):
        '''Abstract method to fit the model.
        
        Parameters
        ----------
        model : xe.models.EOF
            A EOF model solution.

        '''
        self._model = model

        # Here follows the implementation to fit the model
        # ATTRIBUTES TO BE DEFINED:
        self._rotation_matrix = None
        self._explained_variance = None
        self._explained_variance_ratio = None
        self._components = None
        self._scores = None
    
    @abstractmethod
    def transform(self):
        raise NotImplementedError
    
    @abstractmethod
    def inverse_transform(self):
        raise NotImplementedError
    
    def explained_variance(self):
        return self._explained_variance
    
    def explained_variance_ratio(self):
        return self._explained_variance_ratio
    
    def components(self):
        return self._model.stacker.inverse_transform_components(self._components)  #type: ignore
    
    def scores(self):
        return self._model.stacker.inverse_transform_scores(self._scores)  #type: ignore

    def _get_rotation_matrix(self, inverse_transpose : bool = False) -> xr.DataArray:
        '''Return the rotation matrix.

        Parameters
        ----------
        inverse_transpose : boolean
            If True, return the inverse transposed of the rotation matrix.
            For orthogonal rotations (Varimax) the inverse transpose equals
            the rotation matrix itself. For oblique rotations (Promax), it
            will be different in general (the default is False).

        Returns
        -------
        rotation_matrix : xr.DataArray

        '''
        R = self._rotation_matrix
        if inverse_transpose and self._params['power'] > 1:
            R = xr.apply_ufunc(
                np.linalg.pinv,
                R,
                input_core_dims=[['mode','mode1']],
                output_core_dims=[['mode','mode1']]
            )
            R = R.conj().T
        return R  # type: ignore

