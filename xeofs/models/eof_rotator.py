from datetime import datetime
import numpy as np
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from typing import List

from .eof import EOF, ComplexEOF
from ..data_container.eof_rotator_data_container import EOFRotatorDataContainer, ComplexEOFRotatorDataContainer

from ..utils.rotation import promax
from ..utils.data_types import DataArray, AnyDataObject

from typing import TypeVar
from .._version import __version__

Model = TypeVar('Model', EOF, ComplexEOF)


class EOFRotator(EOF):
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

    def __init__(
            self,
            n_modes: int = 10,
            power: int = 1,
            max_iter: int = 1000,
            rtol: float = 1e-8,
        ):
        # Define model parameters
        self._params = {
            'n_modes': n_modes,
            'power': power,
            'max_iter': max_iter,
            'rtol': rtol,
        }
        
        # Define analysis-relevant meta data
        self.attrs = {'model': 'Rotated EOF analysis'}
        self.attrs.update(self._params)
        self.attrs.update({
            'software': 'xeofs',
            'version': __version__,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        # Initialize the DataContainer to store the results
        self.data: EOFRotatorDataContainer = EOFRotatorDataContainer()

    def fit(self, model):
        '''Fit the model.
        
        Parameters
        ----------
        model : xe.models.EOF
            A EOF model solution.
            
        '''
        self.model = model
        self.preprocessor = model.preprocessor

        n_modes = self._params.get('n_modes')
        power = self._params.get('power')
        max_iter = self._params.get('max_iter')
        rtol = self._params.get('rtol')

        # Select modes to rotate
        components = model.data.components.sel(mode=slice(1, n_modes))
        expvar = model.data.explained_variance.sel(mode=slice(1, n_modes))

        # Rotate loadings
        loadings = components * np.sqrt(expvar)
        rot_loadings, rot_matrix, phi_matrix =  xr.apply_ufunc(
            promax,
            loadings,
            power,
            input_core_dims=[['feature', 'mode'], []],
            output_core_dims=[['feature', 'mode'], ['mode_m', 'mode_n'], ['mode_m', 'mode_n']],
            kwargs={'max_iter': max_iter, 'rtol': rtol},
            dask='allowed'
        )
        # Assign coordinates to the rotation/correlation matrices
        rot_matrix = rot_matrix.assign_coords(
            mode_m=np.arange(1, rot_matrix.mode_m.size+1),
            mode_n=np.arange(1, rot_matrix.mode_n.size+1),
        )
        phi_matrix = phi_matrix.assign_coords(
            mode_m=np.arange(1, phi_matrix.mode_m.size+1),
            mode_n=np.arange(1, phi_matrix.mode_n.size+1),
        )
        
        # Reorder according to variance
        expvar = (abs(rot_loadings)**2).sum('feature')
        # NOTE: For delayed objects, the index must be computed.
        # NOTE: The index must be computed before sorting since argsort is not (yet) implemented in dask
        idx_sort = expvar.compute().argsort()[::-1]
        idx_sort.coords.update(expvar.coords)
        
        expvar = expvar.isel(mode=idx_sort.values).assign_coords(mode=expvar.mode)
        rot_loadings = rot_loadings.isel(mode=idx_sort.values).assign_coords(mode=rot_loadings.mode)

        # Normalize loadings
        rot_components = rot_loadings / np.sqrt(expvar)

        # Rotate scores
        scores = model.data.scores.sel(mode=slice(1,n_modes))
        RinvT = self._compute_rot_mat_inv_trans(rot_matrix, input_dims=('mode_m', 'mode_n'))
        # Rename dimension mode to ensure that dot product has dimensions (sample x mode) as output
        scores = scores.rename({'mode':'mode_m'})
        RinvT = RinvT.rename({'mode_n':'mode'})
        scores = xr.dot(scores, RinvT, dims='mode_m') 

        # Reorder according to variance
        scores = scores.isel(mode=idx_sort.values).assign_coords(mode=scores.mode)

        # Ensure consitent signs for deterministic output
        idx_max_value = abs(rot_loadings).argmax('feature').compute()
        modes_sign = xr.apply_ufunc(np.sign, rot_loadings.isel(feature=idx_max_value), dask='allowed')
        # Drop all dimensions except 'mode' so that the index is clean
        for dim, coords in modes_sign.coords.items():
            if dim != 'mode':
                modes_sign = modes_sign.drop(dim)
        rot_components = rot_components * modes_sign
        scores = scores * modes_sign

        # Create the data container
        self.data.set_data(
            input_data=model.data.input_data,
            components=rot_components,
            scores=scores,
            explained_variance=expvar,
            total_variance=model.data.total_variance,
            idx_modes_sorted=idx_sort,
            rotation_matrix=rot_matrix,
            phi_matrix=phi_matrix,
            modes_sign=modes_sign,
        )
        # Assign analysis-relevant meta data
        self.data.set_attrs(self.attrs)


    def transform(self, data: AnyDataObject) -> DataArray:

        n_modes = self._params['n_modes']

        svals = self.model.data.singular_values.sel(mode=slice(1, self._params['n_modes']))
        # Select the (non-rotated) singular vectors of the first dataset
        components = self.model.data.components.sel(mode=slice(1, n_modes))

        # Preprocess the data
        da: DataArray = self.preprocessor.transform(data)

        # Compute non-rotated scores by project the data onto non-rotated components
        projections = xr.dot(da, components) / svals
        projections.name = 'scores'

        # Rotate the scores
        R = self.data.rotation_matrix
        RinvT = self._compute_rot_mat_inv_trans(R, input_dims=('mode_m', 'mode_n'))
        projections = projections.rename({'mode':'mode_m'})
        RinvT = RinvT.rename({'mode_n':'mode'})
        projections = xr.dot(projections, RinvT, dims='mode_m')
        # Reorder according to variance
        # this must be done in one line: i) select modes according to their variance, ii) replace coords with modes from 1 ... n
        projections = projections.isel(mode=self.data.idx_modes_sorted.values).assign_coords(mode=projections.mode)

        # Adapt the sign of the scores
        projections = projections * self.data.modes_sign
        
        # Unstack the projections
        projections = self.preprocessor.inverse_transform_scores(projections)
        return projections      
    
    def _compute_rot_mat_inv_trans(self, rotation_matrix, input_dims) -> DataArray:
        '''Compute the inverse transpose of the rotation matrix.

        For orthogonal rotations (e.g., Varimax), the inverse transpose is equivalent 
        to the rotation matrix itself. For oblique rotations (e.g., Promax), the simplification
        does not hold.
        
        Returns
        -------
        rotation_matrix : xr.DataArray

        '''
        if self._params['power'] > 1:
            # inverse matrix
            rotation_matrix = xr.apply_ufunc(
                np.linalg.inv,
                rotation_matrix,
                input_core_dims=[(input_dims)],
                output_core_dims=[(input_dims[::-1])],
                vectorize=False,
                dask='allowed',
            )
            # transpose matrix
            rotation_matrix = rotation_matrix.conj().transpose(*input_dims)
        return rotation_matrix 


class ComplexEOFRotator(EOFRotator, ComplexEOF):
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
    
        # Initialize the DataContainer to store the results
        self.data: ComplexEOFRotatorDataContainer = ComplexEOFRotatorDataContainer()

    def transform(self, data: AnyDataObject):
        # Here we make use of the Method Resolution Order (MRO) to call the
        # transform method of the first class in the MRO after `EOFRotator` 
        # that has a transform method. In this case it will be `ComplexEOF`,
        # which will raise an error because it does not have a transform method.
        super(EOFRotator, self).transform(data)
