from abc import ABC, abstractmethod

import numpy as np
import scipy as sc
import xarray as xr
from typing import Optional, Union, List, Tuple

from ._base_model import EOF, ComplexEOF
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
    
    def fit(self, model: EOF | ComplexEOF):
        '''Fit the model.
        
        Parameters
        ----------
        model : xe.models.EOF
            A EOF model solution.
            
        '''
        self._model = model

        n_rot = self._params['n_rot']
        power = self._params['power']
        max_iter = self._params['max_iter']
        rtol = self._params['rtol']

        # Select modes to rotate
        components = self._model._components.sel(mode=slice(1, n_rot))
        expvar = self._model._explained_variance.sel(mode=slice(1, n_rot))

        # Rotate loadings
        loadings = components * np.sqrt(expvar)
        rot_loadings, rot_matrix, Phi =  xr.apply_ufunc(
            promax,
            loadings,
            power,
            input_core_dims=[['feature', 'mode'], []],
            output_core_dims=[['feature', 'mode'], ['mode', 'mode1'], ['mode', 'mode1']],
            kwargs={'max_iter': max_iter, 'rtol': rtol},
        )
        self._rotation_matrix = rot_matrix
        
        # Reorder according to variance
        expvar = (abs(rot_loadings)**2).sum('feature')
        idx_sort = expvar.argsort().values[::-1]
        expvar = expvar.isel(mode=idx_sort).assign_coords(mode=expvar.mode)
        rot_loadings = rot_loadings.isel(mode=idx_sort).assign_coords(mode=rot_loadings.mode)
        self._idx_expvar = idx_sort

        # Explained variance
        self._explained_variance = expvar
        self._explained_variance_ratio = expvar / self._model._total_variance

        # Normalize loadings
        rot_components = rot_loadings / np.sqrt(expvar)
        self._components = rot_components

        # Rotate scores
        scores = self._model._scores.sel(mode=slice(1,n_rot))
        R = self._get_rotation_matrix(inverse_transpose=True)
        scores = xr.dot(scores, R, dims='mode1')

        # Reorder according to variance
        scores = scores.isel(mode=idx_sort).assign_coords(mode=scores.mode)
        self._scores = scores

    
    def inverse_transform(self, mode: int | List[int] | slice = slice(None)):
        dof = self._model.data.shape[0] - 1

        components = self._components
        scores = self._scores * np.sqrt(self._explained_variance * dof)  # type: ignore

        components = components.sel(mode=mode)  # type: ignore
        scores = scores.sel(mode=mode)
        Xrec = xr.dot(scores, components.conj(), dims='mode')

        Xrec = self._model.stacker.inverse_transform_data(Xrec)
        Xrec = self._model.scaler.inverse_transform(Xrec)  # type: ignore
        
        return Xrec
    
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

