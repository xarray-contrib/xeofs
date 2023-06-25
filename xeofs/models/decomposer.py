import numpy as np
import xarray as xr
from dask.array import Array as DaskArray
from sklearn.utils.extmath import randomized_svd as svd
from scipy.sparse.linalg import svds as complex_svd
from dask.array.linalg import svd_compressed as dask_svd


class Decomposer():
    def __init__(self, n_components=100, allow_complex=False, n_iter=5, random_state=None, verbose=False):
        self.params = {
            'n_components': n_components,
            'allow_complex': allow_complex,
            'use_dask': False,
            'n_iter': n_iter,
            'random_state': random_state,
            'verbose': verbose,
        }

    def fit(self, X):
        svd_kwargs = {}
        
        if isinstance(X.data, DaskArray):
            self.params['use_dask'] = True
            
        if (not self.params['allow_complex']) and (not self.params['use_dask']):
            svd_kwargs.update({
                'n_components': self.params['n_components'],
                'random_state': self.params['random_state']
            })

            U, s, VT = xr.apply_ufunc(
                svd,
                X,
                kwargs=svd_kwargs,
                input_core_dims=[['sample', 'feature']],
                output_core_dims=[['sample', 'mode'], ['mode'], ['mode', 'feature']],
            )

        elif (self.params['allow_complex']) and (not self.params['use_dask']):
            # Scipy sparse version
            svd_kwargs.update({
                'solver': 'lobpcg',
                'k': self.params['n_components'],
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

        elif (not self.params['allow_complex']) and (self.params['use_dask']):
            svd_kwargs.update({
                'k': self.params['n_components']
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

        self.scores_ = U
        self.singular_values_ = s
        self.components_ = VT

        
