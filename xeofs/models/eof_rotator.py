import numpy as np
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from typing import List

from ._base_rotator import _BaseRotator
from .eof import EOF, ComplexEOF

from ..utils.rotation import promax
from ..utils.data_types import DataArray, Dataset, XarrayData, DataArrayList


class EOFRotator(_BaseRotator):
    '''Rotate a solution obtained from ``xe.models.EOF``.

    Parameters
    ----------
    n_modes : int
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
        super().__init__(**kwargs)
        self.attrs.update({'model': 'Rotated EOF analysis'})

    def fit(self, model: EOF | ComplexEOF):
        '''Fit the model.
        
        Parameters
        ----------
        model : xe.models.EOF
            A EOF model solution.
            
        '''
        self._model = model

        n_modes = self._params.get('n_modes')
        power = self._params.get('power')
        max_iter = self._params.get('max_iter')
        rtol = self._params.get('rtol')

        # Select modes to rotate
        components = self._model._components.sel(mode=slice(1, n_modes))
        expvar = self._model._explained_variance.sel(mode=slice(1, n_modes))

        # Rotate loadings
        loadings = components * np.sqrt(expvar)
        rot_loadings, rot_matrix, Phi =  xr.apply_ufunc(
            promax,
            loadings,
            power,
            input_core_dims=[['feature', 'mode'], []],
            output_core_dims=[['feature', 'mode'], ['mode', 'mode1'], ['mode', 'mode1']],
            kwargs={'max_iter': max_iter, 'rtol': rtol},
            dask='allowed'
        )
        self._rotation_matrix = rot_matrix
        
        # Reorder according to variance
        expvar = (abs(rot_loadings)**2).sum('feature')
        # NOTE: For delayed objects, the index must be computed. .values will rensure that the index is computed
        # NOTE: The index must be computed before sorting since argsort is not (yet) implemented in dask
        idx_sort = expvar.values.argsort()[::-1]
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
        scores = self._model._scores.sel(mode=slice(1,n_modes))
        R = self._get_rotation_matrix(inverse_transpose=True)
        scores = xr.dot(scores, R, dims='mode1')

        # Reorder according to variance
        scores = scores.isel(mode=idx_sort).assign_coords(mode=scores.mode)
        self._scores = scores

        # Assign analysis-relevant meta data
        self._assign_meta_data()

    def transform(self, data: XarrayData | DataArrayList) -> XarrayData | DataArrayList:

        n_modes = self._params['n_modes']
        svals = self._model._singular_values.sel(mode=slice(1, self._params['n_modes']))
        components = self._model._components.sel(mode=slice(1, n_modes))

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
    
    def inverse_transform(self, mode: int | List[int] | slice = slice(None)):
        dof = self._model.data.shape[0] - 1  # type: ignore

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
    
    def compute(self, verbose: bool = False):
        '''Compute and load the rotated solution.
        
        Parameters
        ----------
        verbose : bool
            If True, print information about the computation process.
            
        '''

        self._model.compute(verbose=verbose)
        if verbose:
            with ProgressBar():
                print('Computing ROTATED MODEL...')
                print('-'*80)
                print('Explained variance...')
                self._explained_variance = self._explained_variance.compute()
                print('Explained variance ratio...')
                self._explained_variance_ratio = self._explained_variance_ratio.compute()
                print('Components...')
                self._components = self._components.compute()
                print('Rotation matrix...')
                self._rotation_matrix = self._rotation_matrix.compute()
                print('Scores...')
                self._scores = self._scores.compute()

        else:
            self._explained_variance = self._explained_variance.compute()
            self._explained_variance_ratio = self._explained_variance_ratio.compute()
            self._components = self._components.compute()
            self._rotation_matrix = self._rotation_matrix.compute()
            self._scores = self._scores.compute()


    def _assign_meta_data(self):
        '''Assign analysis-relevant meta data.'''
        # Attributes of fitted model
        attrs = self._model.attrs.copy()  # type: ignore
        # Include meta data of the rotation
        attrs.update(self.attrs)
        self._explained_variance.attrs.update(attrs)  # type: ignore
        self._explained_variance_ratio.attrs.update(attrs)  # type: ignore
        self._components.attrs.update(attrs)  # type: ignore
        self._scores.attrs.update(attrs)  # type: ignore

 
class ComplexEOFRotator(EOFRotator):
    '''Rotate a solution obtained from ``xe.models.ComplexEOF``.

    Parameters
    ----------
    n_modes : int
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
        super().__init__(**kwargs)
        self.attrs.update({'model': 'Rotated Complex EOF analysis'})

    def transform(self, data: XarrayData | DataArrayList):
        raise NotImplementedError('Complex EOF does not support transform.')

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
        comps = xr.apply_ufunc(np.angle, self._components, dask='allowed', keep_attrs=True)
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
        scores = xr.apply_ufunc(np.angle, self._scores, dask='allowed', keep_attrs=True)
        scores.name = 'phase'
        scores = self._model.stacker.inverse_transform_scores(scores)
        return scores
