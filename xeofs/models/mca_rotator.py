from datetime import datetime
import numpy as np
import xarray as xr
from typing import List

from .mca import MCA, ComplexMCA
from ..utils.rotation import promax
from ..utils.data_types import DataArray
from ..data_container.mca_rotator_data_container import MCARotatorDataContainer, ComplexMCARotatorDataContainer
from .._version import __version__


class MCARotator(MCA):
    '''Rotate a solution obtained from ``xe.models.MCA``.
    
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
    squared_loadings : bool, default=False
        Determines the method of constructing the combined vectors of loadings. If set to True, the combined 
        vectors are loaded with the singular values ("squared loadings"), conserving the squared covariance 
        under rotation. This allows for estimation of mode importance after rotation. If set to False, 
        follows the Cheng & Dunkerton method [1]_ of loading with the square root of singular values.
    
    References
    ----------
    .. [1] Cheng, X., Dunkerton, T.J., 1995. Orthogonal Rotation of Spatial Patterns Derived from Singular Value Decomposition Analysis. J. Climate 8, 2631–2643. https://doi.org/10.1175/1520-0442(1995)008<2631:OROSPD>2.0.CO;2

    '''

    def __init__(
            self,
            n_modes: int = 10,
            power: int = 1,
            max_iter: int = 1000,
            rtol: float = 1e-8,
            squared_loadings: bool = False
        ):
        # Define model parameters
        self._params = {
            'n_modes': n_modes,
            'power': power,
            'max_iter': max_iter,
            'rtol': rtol,
            'squared_loadings': squared_loadings,
        }
        
        # Define analysis-relevant meta data
        self.attrs = {'model': 'Rotated MCA'}
        self.attrs.update(self._params)
        self.attrs.update({
            'software': 'xeofs',
            'version': __version__,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    def _compute_rot_mat_inv_trans(self, rotation_matrix) -> xr.DataArray:
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
                np.linalg.pinv,
                rotation_matrix,
                input_core_dims=[['mode','mode1']],
                output_core_dims=[['mode','mode1']]
            )
            # transpose matrix
            rotation_matrix = rotation_matrix.conj().T
        return rotation_matrix 

    @staticmethod
    def _create_data_container(**kwargs) -> MCARotatorDataContainer:
        '''Create a data container for the rotated solution.

        '''
        return MCARotatorDataContainer(**kwargs)  # type: ignore

    def fit(self, model: MCA | ComplexMCA):
        '''Fit the model.
        
        Parameters
        ----------
        model : xe.models.MCA
            A MCA model solution.
            
        '''
        self.model = model
        self.preprocessor1 = model.preprocessor1
        self.preprocessor2 = model.preprocessor2
        
        n_modes = self._params['n_modes']
        power = self._params['power']
        max_iter = self._params['max_iter']
        rtol = self._params['rtol']
        use_squared_loadings = self._params['squared_loadings']


        # Construct the combined vector of loadings
        # NOTE: In the methodology used by Cheng & Dunkerton (CD), the combined vectors are "loaded" or weighted 
        # with the square root of the singular values, akin to what is done in standard Varimax rotation. This method 
        # ensures that the total amount of covariance is conserved during the rotation process.
        # However, in Maximum Covariance Analysis (MCA), the focus is usually on the squared covariance to determine
        # the importance of a given mode. The approach adopted by CD does not preserve the squared covariance under
        # rotation, making it impossible to estimate the importance of modes post-rotation.
        # To resolve this issue, one possible workaround is to rotate the singular vectors that have been "loaded"
        # or weighted with the singular values ("squared loadings"), as opposed to the square root of the singular values.
        # In doing so, the squared covariance remains conserved under rotation, allowing for the estimation of the 
        # modes' importance. 
        norm1 = self.model.data.norm1.sel(mode=slice(1, n_modes))
        norm2 = self.model.data.norm2.sel(mode=slice(1, n_modes))
        if use_squared_loadings:
            # Squared loadings approach conserving squared covariance
            scaling = norm1 * norm2
        else:
            # Cheng & Dunkerton approach conserving covariance
            scaling = np.sqrt(norm1 * norm2)


        comps1 = self.model.data.components1.sel(mode=slice(1, n_modes))
        comps2 = self.model.data.components2.sel(mode=slice(1, n_modes))
        loadings = xr.concat([comps1, comps2], dim='feature') * scaling

        # Rotate loadings
        rot_loadings, rot_matrix, phi_matrix =  xr.apply_ufunc(
            promax,
            loadings,
            power,
            input_core_dims=[['feature', 'mode'], []],
            output_core_dims=[['feature', 'mode'], ['mode', 'mode1'], ['mode', 'mode1']],
            kwargs={'max_iter': max_iter, 'rtol': rtol},
            dask='allowed'
        )
        
        # Rotated (loaded) singular vectors
        comps1_rot = rot_loadings.isel(feature=slice(0, comps1.coords['feature'].size))
        comps2_rot = rot_loadings.isel(feature=slice(comps1.coords['feature'].size, None))

        # Normalization factor of singular vectors
        norm1_rot = xr.apply_ufunc(np.linalg.norm, comps1_rot, input_core_dims=[['feature']], output_core_dims=[[]], kwargs={'axis': -1}, dask='allowed')
        norm2_rot = xr.apply_ufunc(np.linalg.norm, comps2_rot, input_core_dims=[['feature']], output_core_dims=[[]], kwargs={'axis': -1}, dask='allowed')

        # Rotated (normalized) singular vectors
        comps1_rot = comps1_rot / norm1_rot
        comps2_rot = comps2_rot / norm2_rot

        # Remove the squaring introduced by the squared loadings approach
        if use_squared_loadings:
            norm1_rot = norm1_rot ** 0.5
            norm2_rot = norm2_rot ** 0.5
        
        # norm1 * norm2 = "singular values"
        squared_covariance = (norm1_rot * norm2_rot)**2

        # Reorder according to squared covariance
        # NOTE: For delayed objects, the index must be computed.
        # NOTE: The index must be computed before sorting since argsort is not (yet) implemented in dask
        idx_modes_sorted = squared_covariance.compute().argsort()[::-1]
        idx_modes_sorted.coords.update(squared_covariance.coords)
        
        squared_covariance = squared_covariance.isel(mode=idx_modes_sorted).assign_coords(mode=squared_covariance.mode)

        norm1_rot = norm1_rot.isel(mode=idx_modes_sorted).assign_coords(mode=norm1_rot.mode)
        norm2_rot = norm2_rot.isel(mode=idx_modes_sorted).assign_coords(mode=norm2_rot.mode)
        
        comps_rot1 = comps1_rot.isel(mode=idx_modes_sorted).assign_coords(mode=comps1_rot.mode)
        comps_rot2 = comps2_rot.isel(mode=idx_modes_sorted).assign_coords(mode=comps2_rot.mode)

          # Rotate scores using rotation matrix
        scores1 = self.model.data.scores1.sel(mode=slice(1,n_modes))
        scores2 = self.model.data.scores2.sel(mode=slice(1,n_modes))
        R = self._compute_rot_mat_inv_trans(rot_matrix)

        # The following renaming is necessary to ensure that the output dimension is `mode`
        scores1 = xr.dot(scores1, R, dims='mode')
        scores2 = xr.dot(scores2, R, dims='mode')
        scores1 = scores1.assign_coords({'mode1': np.arange(1, n_modes + 1)})
        scores2 = scores2.assign_coords({'mode1': np.arange(1, n_modes + 1)})
        scores1 = scores1.rename({'mode1': 'mode'})
        scores2 = scores2.rename({'mode1': 'mode'})

        # Reorder scores according to variance
        scores1 = scores1.isel(mode=idx_modes_sorted).assign_coords(mode=scores1.mode)
        scores2 = scores2.isel(mode=idx_modes_sorted).assign_coords(mode=scores2.mode)
        

        # Ensure consitent signs for deterministic output
        idx_max_value = abs(rot_loadings).argmax('feature').compute()
        modes_sign = xr.apply_ufunc(np.sign, rot_loadings.isel(feature=idx_max_value), dask='allowed')
        # Drop all dimensions except 'mode' so that the index is clean
        for dim, coords in modes_sign.coords.items():
            if dim != 'mode':
                modes_sign = modes_sign.drop(dim)
        comps_rot1 = comps_rot1 * modes_sign
        comps_rot2 = comps_rot2 * modes_sign
        scores1 = scores2 * modes_sign
        scores2 = scores2 * modes_sign

        # Create data container
        self.data = self._create_data_container(
            input_data1=self.model.data.input_data1,
            input_data2=self.model.data.input_data2,
            components1=comps_rot1,
            components2=comps_rot2,
            scores1=scores1,
            scores2=scores2,
            squared_covariance=squared_covariance,
            total_squared_covariance=self.model.data.total_squared_covariance,
            idx_modes_sorted=idx_modes_sorted,
            norm1=norm1_rot,
            norm2=norm2_rot,
            rotation_matrix=rot_matrix,
            phi_matrix=phi_matrix,
            modes_sign=modes_sign,
        )
        
        # Assign analysis-relevant meta data
        self.data.set_attrs(self.attrs)

    def transform(self, **kwargs) -> DataArray | List[DataArray]:
        '''Project new "unseen" data onto the rotated singular vectors.

        Parameters
        ----------
        data1 : DataArray | Dataset | DataArraylist
            Data to be projected onto the rotated singular vectors of the first dataset.
        data2 : DataArray | Dataset | DataArraylist
            Data to be projected onto the rotated singular vectors of the second dataset.

        Returns
        -------
        DataArray | List[DataArray]
            Projected data.
        
        '''
        # raise error if no data is provided
        if not kwargs:
            raise ValueError('No data provided. Please provide data1 and/or data2.')
    
        n_modes = self._params['n_modes']
        rot_matrix = self.data.rotation_matrix
        rot_matrix = self._compute_rot_mat_inv_trans(rot_matrix)

        results = []

        if 'data1' in kwargs.keys():

            data1 = kwargs['data1']
            # Select the (non-rotated) singular vectors of the first dataset
            comps1 = self.model.data.components1.sel(mode=slice(1, n_modes))
            norm1 = self.data.norm1.sel(mode=slice(1, n_modes))
            
            # Preprocess the data
            data1 = self.preprocessor1.transform(data1)
            
            # Compute non-rotated scores by projecting the data onto non-rotated components
            projections1 = xr.dot(data1, comps1) / norm1
            # Rotate the scores
            projections1 = xr.dot(projections1, rot_matrix, dims='mode1')
            # Reorder according to variance
            projections1 = projections1.isel(mode=self.data.idx_modes_sorted).assign_coords(mode=projections1.mode)
            # Adapt the sign of the scores
            projections1 = projections1 * self.data.modes_sign

            # Unstack the projections
            projections1 = self.preprocessor1.inverse_transform_scores(projections1)

            results.append(projections1)


        if 'data2' in kwargs.keys():

            data2 = kwargs['data2']            
            # Select the (non-rotated) singular vectors of the second dataset
            comps2 = self.model.data.components2.sel(mode=slice(1, n_modes))
            norm2 = self.data.norm2.sel(mode=slice(1, n_modes))
            
            # Preprocess the data
            data2 = self.preprocessor2.transform(data2)
            
            # Compute non-rotated scores by project the data onto non-rotated components
            projections2 = xr.dot(data2, comps2) / norm2
            # Rotate the scores
            projections2 = xr.dot(projections2, rot_matrix, dims='mode1')
            # Reorder according to variance
            projections2 = projections2.isel(mode=self.data.idx_modes_sorted).assign_coords(mode=projections2.mode)
            # Determine the sign of the scores
            projections2 = projections2 * self.data.modes_sign


            # Unstack the projections
            projections2 = self.preprocessor2.inverse_transform_scores(projections2)

            results.append(projections2)
        
        if len(results) == 0:
            raise ValueError('provide at least one of [`data1`, `data2`]')
        elif len(results) == 1:
            return results[0]
        else:
            return results


class ComplexMCARotator(MCARotator, ComplexMCA):
    '''Rotate a solution obtained from ``xe.models.ComplexMCA``.

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
    squared_loadings : bool, default=False
        Determines the method of constructing the combined vectors of loadings. If set to True, the combined 
        vectors are loaded with the singular values ("squared loadings"), conserving the squared covariance 
        under rotation. This allows for estimation of mode importance after rotation. If set to False, 
        follows the Cheng & Dunkerton method [1]_ of loading with the square root of singular values.

    References
    ----------
    .. [1] Cheng, X., Dunkerton, T.J., 1995. Orthogonal Rotation of Spatial Patterns Derived from Singular Value Decomposition Analysis. J. Climate 8, 2631–2643. https://doi.org/10.1175/1520-0442(1995)008<2631:OROSPD>2.0.CO;2

    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attrs.update({'model': 'Complex Rotated MCA'})

    @staticmethod
    def _create_data_container(**kwargs) -> ComplexMCARotatorDataContainer:
        '''Create a data container for the rotated solution.

        '''
        return ComplexMCARotatorDataContainer(**kwargs)

    def transform(self, **kwargs):
        # Here we make use of the Method Resolution Order (MRO) to call the
        # transform method of the first class in the MRO after `MCARotator` 
        # that has a transform method. In this case it will be `ComplexMCA`,
        # which will raise an error because it does not have a transform method.
        super(MCARotator, self).transform(**kwargs)
    
    def homogeneous_patterns(self, **kwargs):
        super(MCARotator, self).homogeneous_patterns(**kwargs)

    def heterogeneous_patterns(self, **kwargs):
        super(MCARotator, self).homogeneous_patterns(**kwargs)