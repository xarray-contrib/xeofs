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

    Rotated MCA [1]_ is an extension of the standard MCA that applies an additional rotation 
    to the computed modes to maximize the variance explained individually by each mode. 
    This rotation method enhances interpretability by distributing the explained variance more 
    evenly among the modes, making it easier to discern patterns within the data.
    
    Parameters
    ----------
    n_modes : int, default=10
        Specify the number of modes to be rotated.
    power : int, default=1
        Set the power for the Promax rotation. A ``power`` value of 1 results 
        in a Varimax rotation.
    max_iter : int, default=1000
        Determine the maximum number of iterations for the computation of the 
        rotation matrix.
    rtol : float, default=1e-8
        Define the relative tolerance required to achieve convergence and 
        terminate the iterative process.
    squared_loadings : bool, default=False
        Specify the method for constructing the combined vectors of loadings. If True, 
        the combined vectors are loaded with the singular values (termed "squared loadings"), 
        conserving the squared covariance under rotation. This allows estimation of mode importance 
        after rotation. If False, the combined vectors are loaded with the square root of the 
        singular values, following the method described by Cheng & Dunkerton [1]_.
        
    References
    ----------
    .. [1] Cheng, X., Dunkerton, T.J., 1995. Orthogonal Rotation of Spatial Patterns Derived from Singular Value Decomposition Analysis. J. Climate 8, 2631–2643. https://doi.org/10.1175/1520-0442(1995)008<2631:OROSPD>2.0.CO;2

    Examples
    --------
    >>> model = MCA(n_modes=5)
    >>> model.fit(da1, da2, dim='time')
    >>> rotator = MCARotator(n_modes=5, power=2)
    >>> rotator.fit(model)
    >>> rotator.components()

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

        # Initialize the DataContainer to hold the rotated solution
        self.data: MCARotatorDataContainer = MCARotatorDataContainer()

    def _compute_rot_mat_inv_trans(self, rotation_matrix, input_dims) -> xr.DataArray:
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
        
        # Rotated (loaded) singular vectors
        comps1_rot = rot_loadings.isel(feature=slice(0, comps1.coords['feature'].size))
        comps2_rot = rot_loadings.isel(feature=slice(comps1.coords['feature'].size, None))

        # Normalization factor of singular vectors
        norm1_rot = xr.apply_ufunc(np.linalg.norm, comps1_rot, input_core_dims=[['feature', 'mode']], output_core_dims=[['mode']], exclude_dims={'feature'}, kwargs={'axis':0}, vectorize=False, dask='allowed')
        norm2_rot = xr.apply_ufunc(np.linalg.norm, comps2_rot, input_core_dims=[['feature', 'mode']], output_core_dims=[['mode']], exclude_dims={'feature'}, kwargs={'axis':0}, vectorize=False, dask='allowed')

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
        
        squared_covariance = squared_covariance.isel(mode=idx_modes_sorted.values).assign_coords(mode=squared_covariance.mode)

        norm1_rot = norm1_rot.isel(mode=idx_modes_sorted.values).assign_coords(mode=norm1_rot.mode)
        norm2_rot = norm2_rot.isel(mode=idx_modes_sorted.values).assign_coords(mode=norm2_rot.mode)
        
        comps1_rot = comps1_rot.isel(mode=idx_modes_sorted.values).assign_coords(mode=comps1_rot.mode)
        comps2_rot = comps2_rot.isel(mode=idx_modes_sorted.values).assign_coords(mode=comps2_rot.mode)

        # Rotate scores using rotation matrix
        scores1 = self.model.data.scores1.sel(mode=slice(1,n_modes))
        scores2 = self.model.data.scores2.sel(mode=slice(1,n_modes))

        RinvT = self._compute_rot_mat_inv_trans(rot_matrix, input_dims=('mode_m', 'mode_n'))
        # Rename dimension mode to ensure that dot product has dimensions (sample x mode) as output
        scores1 = scores1.rename({'mode':'mode_m'})
        scores2 = scores2.rename({'mode':'mode_m'})
        RinvT = RinvT.rename({'mode_n':'mode'})
        scores1_rot = xr.dot(scores1, RinvT, dims='mode_m') 
        scores2_rot = xr.dot(scores2, RinvT, dims='mode_m') 

        # Reorder scores _rotaccording to variance
        scores1_rot = scores1_rot.isel(mode=idx_modes_sorted.values).assign_coords(mode=scores1_rot.mode)
        scores2_rot = scores2_rot.isel(mode=idx_modes_sorted.values).assign_coords(mode=scores2_rot.mode)
        

        # Ensure consitent signs for deterministic output
        idx_max_value = abs(rot_loadings).argmax('feature').compute()
        modes_sign = xr.apply_ufunc(np.sign, rot_loadings.isel(feature=idx_max_value), dask='allowed')
        # Drop all dimensions except 'mode' so that the index is clean
        for dim, coords in modes_sign.coords.items():
            if dim != 'mode':
                modes_sign = modes_sign.drop(dim)
        comps1_rot = comps1_rot * modes_sign
        comps2_rot = comps2_rot * modes_sign
        scores1_rot = scores1_rot * modes_sign
        scores2_rot = scores2_rot * modes_sign

        # Create data container
        self.data.set_data(
            input_data1=self.model.data.input_data1,
            input_data2=self.model.data.input_data2,
            components1=comps1_rot,
            components2=comps2_rot,
            scores1=scores1_rot,
            scores2=scores2_rot,
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
        RinvT = self._compute_rot_mat_inv_trans(rot_matrix, input_dims=('mode_m', 'mode_n'))
        RinvT = RinvT.rename({'mode_n':'mode'})

        results = []

        if 'data1' in kwargs.keys():

            data1 = kwargs['data1']
            # Select the (non-rotated) singular vectors of the first dataset
            comps1 = self.model.data.components1.sel(mode=slice(1, n_modes))
            norm1 = self.model.data.norm1.sel(mode=slice(1, n_modes))
            
            # Preprocess the data
            data1 = self.preprocessor1.transform(data1)
            
            # Compute non-rotated scores by projecting the data onto non-rotated components
            projections1 = xr.dot(data1, comps1) / norm1
            # Rotate the scores
            projections1 = projections1.rename({'mode':'mode_m'})
            projections1 = xr.dot(projections1, RinvT, dims='mode_m')
            # Reorder according to variance
            projections1 = projections1.isel(mode=self.data.idx_modes_sorted.values).assign_coords(mode=projections1.mode)
            # Adapt the sign of the scores
            projections1 = projections1 * self.data.modes_sign

            # Unstack the projections
            projections1 = self.preprocessor1.inverse_transform_scores(projections1)

            results.append(projections1)


        if 'data2' in kwargs.keys():

            data2 = kwargs['data2']            
            # Select the (non-rotated) singular vectors of the second dataset
            comps2 = self.model.data.components2.sel(mode=slice(1, n_modes))
            norm2 = self.model.data.norm2.sel(mode=slice(1, n_modes))
            
            # Preprocess the data
            data2 = self.preprocessor2.transform(data2)
            
            # Compute non-rotated scores by project the data onto non-rotated components
            projections2 = xr.dot(data2, comps2) / norm2
            # Rotate the scores
            projections2 = projections2.rename({'mode':'mode_m'})
            projections2 = xr.dot(projections2, RinvT, dims='mode_m')
            # Reorder according to variance
            projections2 = projections2.isel(mode=self.data.idx_modes_sorted.values).assign_coords(mode=projections2.mode)
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

    Complex Rotated MCA extends the MCA by incorporating both amplitude and phase information 
    using a Hilbert transform prior to performing the MCA and subsequent Varimax or Promax rotation. 
    This adds a further layer of dimensionality to the analysis, allowing for a more nuanced interpretation 
    of complex relationships within the data, particularly useful when analyzing oscillatory data.

    Parameters
    ----------
    n_modes : int, default=10
        Specify the number of modes to be rotated.
    power : int, default=1
        Set the power for the Promax rotation. A ``power`` value of 1 results 
        in a Varimax rotation.
    max_iter : int, default=1000
        Determine the maximum number of iterations for the computation of the 
        rotation matrix.
    rtol : float, default=1e-8
        Define the relative tolerance required to achieve convergence and 
        terminate the iterative process.
    squared_loadings : bool, default=False
        Specify the method for constructing the combined vectors of loadings. If True, 
        the combined vectors are loaded with the singular values (termed "squared loadings"), 
        conserving the squared covariance under rotation. This allows estimation of mode importance 
        after rotation. If False, the combined vectors are loaded with the square root of the 
        singular values, following the method described by Cheng & Dunkerton [1]_.
        
    References
    ----------
    .. [1] Cheng, X., Dunkerton, T.J., 1995. Orthogonal Rotation of Spatial Patterns Derived from Singular Value Decomposition Analysis. J. Climate 8, 2631–2643. https://doi.org/10.1175/1520-0442(1995)008<2631:OROSPD>2.0.CO;2
    .. [2] Elipot, S., Frajka-Williams, E., Hughes, C.W., Olhede, S., Lankhorst, M., 2017. Observed Basin-Scale Response of the North Atlantic Meridional Overturning Circulation to Wind Stress Forcing. Journal of Climate 30, 2029–2054. https://doi.org/10.1175/JCLI-D-16-0664.1
    .. [3] Rieger, N., Corral, Á., Olmedo, E., Turiel, A., 2021. Lagged Teleconnections of Climate Variables Identified via Complex Rotated Maximum Covariance Analysis. Journal of Climate 34, 9861–9878. https://doi.org/10.1175/JCLI-D-21-0244.1

    
    Examples
    --------
    >>> model = ComplexMCA(n_modes=5)
    >>> model.fit(da1, da2, dim='time')
    >>> rotator = ComplexMCARotator(n_modes=5, power=2)
    >>> rotator.fit(model)
    >>> rotator.components()

    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attrs.update({'model': 'Complex Rotated MCA'})

        # Initialize the DataContainer to hold the rotated solution
        self.data: ComplexMCARotatorDataContainer = ComplexMCARotatorDataContainer()

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