import numpy as np
import xarray as xr
from dask.array import Array as DaskArray  #type: ignore
from sklearn.utils.extmath import randomized_svd as svd
from scipy.sparse.linalg import svds as complex_svd  #type: ignore
from dask.array.linalg import svd_compressed as dask_svd


class Decomposer():
    '''Decomposes a data object using Singular Value Decomposition (SVD).
     
    The data object will be decomposed into its components, scores and singular values.

    Parameters
    ----------
    n_modes : int
        Number of components to be computed.
    n_iter : int
        Number of iterations for the SVD algorithm.
    random_state : int
        Random seed for the SVD algorithm.
    verbose : bool
        If True, print information about the SVD algorithm.
    
    '''
    def __init__(self, n_modes=100, n_iter=5, random_state=None, verbose=False):
        self.params = {
            'n_modes': n_modes,
            'n_iter': n_iter,
            'random_state': random_state,
            'verbose': verbose,
        }

    def fit(self, X):
        svd_kwargs = {}
        
        is_dask = True if isinstance(X.data, DaskArray) else False
        is_complex = True if np.iscomplexobj(X.data) else False
            
        if (not is_complex) and (not is_dask):
            svd_kwargs.update({
                'n_components': self.params['n_modes'],
                'random_state': self.params['random_state']
            })

            U, s, VT = xr.apply_ufunc(
                svd,
                X,
                kwargs=svd_kwargs,
                input_core_dims=[['sample', 'feature']],
                output_core_dims=[['sample', 'mode'], ['mode'], ['mode', 'feature']],
            )

        elif is_complex and (not is_dask):
            # Scipy sparse version
            svd_kwargs.update({
                'solver': 'lobpcg',
                'k': self.params['n_modes'],
            })
            U, s, VT = xr.apply_ufunc(
                complex_svd,
                X,
                kwargs=svd_kwargs,
                input_core_dims=[['sample', 'feature']],
                output_core_dims=[['sample', 'mode'], ['mode'], ['mode', 'feature']],
            )
            idx_sort = np.argsort(s)[::-1]
            U = U[:, idx_sort]
            s = s[idx_sort]
            VT = VT[idx_sort, :]

        elif (not is_complex) and is_dask:
            svd_kwargs.update({
                'k': self.params['n_modes']
            })
            U, s, VT = xr.apply_ufunc(
                dask_svd,
                X,
                kwargs=svd_kwargs,
                input_core_dims=[['sample', 'feature']],
                output_core_dims=[['sample', 'mode'], ['mode'], ['mode', 'feature']],
                dask='allowed'
            ) 
        else:
            err_msg = (
                'Complex data together with dask is currently not implemented. See dask issue 7639 '
                'https://github.com/dask/dask/issues/7639'
            )
            raise NotImplementedError(err_msg)
        
        U = U.assign_coords(mode=range(1, U.mode.size + 1))
        s = s.assign_coords(mode=range(1, U.mode.size + 1))
        VT = VT.assign_coords(mode=range(1, U.mode.size + 1))

        U.name = 'scores'
        s.name = 'singular_values'
        VT.name = 'components'

        # Flip signs of components to ensure deterministic output    
        idx_sign = abs(VT).argmax('feature').compute()
        flip_signs = np.sign(VT.isel(feature=idx_sign))
        flip_signs = flip_signs.compute()
        # Drop all dimensions except 'mode' so that the index is clean
        for dim, coords in flip_signs.coords.items():
            if dim != 'mode':
                flip_signs = flip_signs.drop(dim)
        VT *= flip_signs
        U *= flip_signs

        self.scores_ = U
        self.singular_values_ = s
        self.components_ = VT.conj().transpose('feature', 'mode')


        
class CrossDecomposer(Decomposer):
    '''Decomposes two data objects based on their cross-covariance matrix.

    The data objects will be decomposed into left and right singular vectors components and their corresponding
    singular values.

    Parameters
    ----------
    n_modes : int
        Number of components to be computed.
    n_iter : int
        Number of iterations for the SVD algorithm.
    random_state : int
        Random seed for the SVD algorithm.
    verbose : bool
        If True, print information about the SVD algorithm.
    
    '''
    def fit(self, X1, X2):
        # Cannot fit data with different number of samples
        if X1.sample.size != X2.sample.size:
            raise ValueError('The two data objects must have the same number of samples.')
    
        # Rename feature and associated dimensions of data objects to avoid conflicts
        feature_dims_temp1 = {dim: dim + '1' for dim in X1.coords['feature'].coords.keys()}
        feature_dims_temp2 = {dim: dim + '2' for dim in X2.coords['feature'].coords.keys()}
        for old, new in feature_dims_temp1.items():
            X1 = X1.rename({old: new})
        for old, new in feature_dims_temp2.items():
            X2 = X2.rename({old: new})

        # Compute cross-covariance matrix
        # Assuming that X1 and X2 are centered
        cov_matrix = xr.dot(X1.conj(), X2, dims='sample') / (X1.sample.size - 1)

        # Compute (squared) total variance
        self.total_covariance_ = (abs(cov_matrix)).sum().compute()
        self.total_squared_covariance_ = (abs(cov_matrix)**2).sum().compute()

        is_dask = True if isinstance(cov_matrix.data, DaskArray) else False
        is_complex = True if np.iscomplexobj(cov_matrix.data) else False
                        
        svd_kwargs = {}
        if (not is_complex) and (not is_dask):
            svd_kwargs.update({
                'n_components': self.params['n_modes'],
                'random_state': self.params['random_state']
            })

            U, s, VT = xr.apply_ufunc(
                svd,
                cov_matrix,
                kwargs=svd_kwargs,
                input_core_dims=[['feature1', 'feature2']],
                output_core_dims=[['feature1', 'mode'], ['mode'], ['mode', 'feature2']],
            )

        elif (is_complex) and (not is_dask):
            # Scipy sparse version
            svd_kwargs.update({
                'solver': 'lobpcg',
                'k': self.params['n_modes'],
            })
            U, s, VT = xr.apply_ufunc(
                complex_svd,
                cov_matrix,
                kwargs=svd_kwargs,
                input_core_dims=[['feature1', 'feature2']],
                output_core_dims=[['feature1', 'mode'], ['mode'], ['mode', 'feature2']],
            )
            idx_sort = np.argsort(s)[::-1]
            U = U[:, idx_sort]
            s = s[idx_sort]
            VT = VT[idx_sort, :]

        elif (not is_complex) and (is_dask):
            svd_kwargs.update({
                'k': self.params['n_modes']
            })
            U, s, VT = xr.apply_ufunc(
                dask_svd,
                cov_matrix,
                kwargs=svd_kwargs,
                input_core_dims=[['feature1', 'feature2']],
                output_core_dims=[['feature1', 'mode'], ['mode'], ['mode', 'feature2']],
                dask='allowed'
            ) 
        else:
            err_msg = (
                'Complex data together with dask is currently not implemented. See dask issue 7639 '
                'https://github.com/dask/dask/issues/7639'
            )
            raise NotImplementedError(err_msg)
        
        U = U.assign_coords(mode=range(1, U.mode.size + 1))
        s = s.assign_coords(mode=range(1, U.mode.size + 1))
        VT = VT.assign_coords(mode=range(1, U.mode.size + 1))

        U.name = 'left_singular_vectors'
        s.name = 'singular_values'
        VT.name = 'right_singular_vectors'

        # Flip signs of components to ensure deterministic output    
        idx_sign = abs(VT).argmax('feature2').compute()
        flip_signs = np.sign(VT.isel(feature2=idx_sign))
        flip_signs = flip_signs.compute()
        # Drop all dimensions except 'mode' so that the index is clean
        # and multiplying will not introduce additional coordinates
        for dim, coords in flip_signs.coords.items():
            if dim != 'mode':
                flip_signs = flip_signs.drop(dim)
        VT *= flip_signs
        U *= flip_signs

        # Rename back to original feature dimensions (remove 1 and 2)
        for old, new in feature_dims_temp1.items():
            U = U.rename({new: old})
        for old, new in feature_dims_temp2.items():
            VT = VT.rename({new: old})

        self.singular_vectors1_ = U
        self.singular_values_ = s
        self.singular_vectors2_ = VT.conj().transpose('feature', 'mode')

