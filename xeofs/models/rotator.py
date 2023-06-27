import numpy as np
import xarray as xr
from typing import Optional, Union, List, Tuple

from ._base_model import EOF
from ._base_rotator import _BaseRotator
from ..utils.rotation import promax


class Rotator(_BaseRotator):
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

    def fit(self, model: EOF):
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
        
        scores = scores.isel(mode=idx_sort).assign_coords(mode=scores.mode)
        self._scores = scores

    def transform(self):
        return super().transform()
    
    def inverse_transform(self):
        return super().inverse_transform()
    
    # def eofs_as_correlation(self) -> Tuple[np.ndarray, np.ndarray]:
    #     '''Correlation coefficients between rotated PCs and data matrix.

    #     Returns
    #     -------
    #     Tuple[np.ndarray, np.ndarry]
    #         Matrices of correlation coefficients and associated
    #         two-sided p-values with features as rows and modes as columns.

    #     '''

    #     # Compute correlation matrix
    #     corr = np.corrcoef(self._model.X, self._pcs, rowvar=False)
    #     corr = corr[:self._model.n_features, self._model.n_features:]
    #     # Compute two-sided p-values
    #     # Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html#r8c6348c62346-1
    #     a = self._model.n_samples / 2 - 1
    #     dist = sc.stats.beta(a, a, loc=-1, scale=2)
    #     pvals = 2 * dist.cdf(-abs(corr))
    #     return corr, pvals

    # def reconstruct_X(
    #     self,
    #     mode : Optional[Union[int, List[int], slice]] = None
    # ) -> np.ndarray:
    #     '''Reconstruct original data field ``X`` using the rotated PCs and EOFs.

    #     If weights were applied, ``X`` will be automatically rescaled.

    #     Parameters
    #     ----------
    #     mode : Optional[Union[int, List[int], slice]]
    #         Mode(s) based on which ``X`` will be reconstructed. If ``mode`` is
    #         an int, a single mode is used. If a list of integers is provided,
    #         use all specified modes for reconstruction. Alternatively, you may
    #         want to select a slice to reconstruct. The first mode is denoted
    #         by 1 (and not by 0). If None then ``X`` is recontructed using all
    #         available modes (the default is None).

    #     Examples
    #     --------

    #     Perform an analysis using some data ``X``:

    #     >>> model = EOF(X, norm=True)
    #     >>> model.solve()

    #     Reconstruct ``X`` using all modes:

    #     >>> model.reconstruct_X()

    #     Reconstruct ``X`` using the first mode only:

    #     >>> model.reconstruct_X(1)

    #     Reconstruct ``X`` using mode 1, 3 and 4:

    #     >>> model.reconstruct_X([1, 3, 4])

    #     Reconstruct ``X`` using all modes up to mode 10 (including):

    #     >>> model.reconstruct_X(slice(10))

    #     Reconstruct ``X`` using every second mode between 4 and 8 (both
    #     including):

    #     >>> model.reconstruct_X(slice(4, 8, 2))


    #     '''
    #     dof = self._model.n_samples - 1
    #     eofs = self._eofs
    #     pcs = self._pcs * np.sqrt(self._explained_variance * dof)
    #     # Select modes to reconstruct X
    #     mode = get_mode_selector(mode)
    #     eofs = eofs[:, mode]
    #     pcs = pcs[:, mode]
    #     Xrec = pcs @ eofs.T
    #     # Unweight and add mean
    #     return (Xrec / self._model._weights) + self._model._X_mean

    # def project_onto_eofs(
    #     self,
    #     X : np.ndarray, scaling : int = 0
    # ) -> np.ndarray:
    #     '''Project new data onto the rotated EOFs.

    #     Parameters
    #     ----------
    #     X : np.ndarray
    #          New data to project. Data must be a 2D matrix.
    #     scaling : [0, 1, 2]
    #         Projections are scaled (i) to be orthonormal (``scaling=0``), (ii)
    #         by the square root of the eigenvalues (``scaling=1``) or (iii) by
    #         the singular values (``scaling=2``). In the latter case, and when
    #         no weights were applied, scaling by the singular values results in
    #         the projections having the unit of the input data
    #         (the default is 0).

    #     '''
    #     dof = self._model.n_samples - 1
    #     svals = np.sqrt(self._model._explained_variance * dof)

    #     # Preprocess new data
    #     try:
    #         X -= self._model._X_mean
    #     except ValueError:
    #         err_msg = (
    #             'New data has invalid feature dimensions and cannot be '
    #             'projected onto EOFs.'
    #         )
    #         raise ValueError(err_msg)
    #     X *= self._model._weights

    #     # Compute non-rotated PCs
    #     pcs = X @ self._model._eofs[:, :self._n_rot] / svals[:self._n_rot]

    #     # Rotate and reorder PCs
    #     R = self._rotation_matrix(inverse_transpose=True)
    #     pcs = pcs @ R
    #     pcs = pcs[:, self._idx_var]

    #     # Apply scaling
    #     if scaling == 0:
    #         return pcs
    #     elif scaling == 1:
    #         return pcs * np.sqrt(self._explained_variance)
    #     elif scaling == 2:
    #         return pcs * np.sqrt(self._explained_variance * dof)
    #     else:
    #         err_msg = (
    #             'Scaling option {:} is not valid but must be one '
    #             'of [0, 1, 2]'
    #         )
    #         err_msg = err_msg.foramt(scaling)
    #         raise ValueError(err_msg)
