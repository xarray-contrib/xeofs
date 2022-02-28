from typing import Iterable, Optional, Union, Tuple, List

import numpy as np
import xarray as xr

from ..models._base_eof import _BaseEOF
from ._dataarray_transformer import _DataArrayTransformer


class MultivariateEOF(_BaseEOF):
    '''EOF analysis of multiple ``xr.DataArray``.

    Parameters
    ----------
    X : List[xr.DataArray]
        Data to be decomposed.
    dim : Optional[Union[str, Iterable[str]]]
        Define the dimension which should considered for maximising variance.
        For most applications in climate science, temporal variance is
        maximised (also known as S-mode EOF analysis) i.e. the time dimension
        should be chosen. If spatial variance should be maximised
        (i.e. T-mode EOF analysis), set e.g. ``dim=['lon', 'lat']``
        (the default is ``time``).
    n_modes : Union[int | None]
        Number of modes to compute. Computing less modes can results in
        performance gains. If None, then the maximum number of modes is
        equivalent to ``min(n_samples, n_features)`` (the default is None).
    norm : bool
        Normalize each feature (e.g. grid cell) by its temporal standard
        deviation (the default is False).
    weights : Optional[Union[List[xr.DatArray], str]]
        Weights to be applied to data (features).


    '''

    def __init__(
        self,
        X: List[xr.DataArray],
        dim: Optional[Union[str, Iterable[str]]] = 'time',
        n_modes : Optional[int] = None,
        norm : bool = False,
        weights : Optional[Union[List[xr.DataArray], str]] = None
    ):

        self._tf = []
        X_transformed = []
        weights_transformed = []

        for x in X:
            wghts = weights
            tf = _DataArrayTransformer()
            # Fit data first, so that _get_coslat_weights can acces transformer
            tf.fit(x, dim=dim)
            if wghts == 'coslat':
                wghts = self._get_coslat_weights(x, tf.dims_samples)
            X_transformed.append(tf.transform(x))
            weights_transformed.append(tf.transform_weights(wghts))
            self._tf.append(tf)

        shapes = [x.shape[1] for x in X_transformed]
        self._multi_idx_features = np.insert(np.cumsum(shapes), 0, 0)
        X = np.concatenate(X_transformed, axis=1)
        weights = np.concatenate(weights_transformed, axis=0) if weights is not None else weights

        super().__init__(
            X=X,
            n_modes=n_modes,
            norm=norm,
            weights=weights
        )
        self._idx_mode = xr.IndexVariable('mode', range(1, self.n_modes + 1))
        self._dim = dim

    def _get_coslat_weights(
        self,
        X : xr.DataArray,
        dims_samples : Iterable[str]
    ) -> xr.DataArray:
        # Find dimension name of latitude
        possible_lat_names = [
            'latitude', 'Latitude', 'lat', 'Lat', 'LATITUDE', 'LAT'
        ]
        idx_lat_dim = np.isin(X.dims, possible_lat_names)
        try:
            lat_dim = np.array(X.dims)[idx_lat_dim][0]
        except IndexError:
            err_msg = (
                'Latitude dimension cannot be found. Please make sure '
                'latitude dimensions is called like one of {:}'
            )
            err_msg = err_msg.format(possible_lat_names)
            raise ValueError(err_msg)
        # Check if latitude is a MultiIndex => not allowed
        if X.coords[lat_dim].dtype not in [np.float_, np.float64, np.float32, np.int_]:
            err_msg = 'MultiIndex as latitude dimensions is not allowed.'
            raise ValueError(err_msg)
        # Compute coslat weights
        weights = np.cos(np.deg2rad(X.coords[lat_dim]))
        weights = np.sqrt(weights.where(weights > 0, 0))
        # Broadcast latitude weights on other feature dimensions
        feature_grid = X.isel({k: 0 for k in dims_samples})
        feature_grid = feature_grid.drop_vars(dims_samples)
        return weights.broadcast_like(feature_grid)

    def singular_values(self) -> xr.DataArray:
        svalues = super().singular_values()
        return xr.DataArray(
            svalues,
            dims=['mode'],
            coords={'mode' : self._idx_mode},
            name='singular_values'
        )

    def explained_variance(self) -> xr.DataArray:
        expvar = super().explained_variance()
        return xr.DataArray(
            expvar,
            dims=['mode'],
            coords={'mode' : self._idx_mode},
            name='explained_variance'
        )

    def explained_variance_ratio(self) -> xr.DataArray:
        expvar = super().explained_variance_ratio()
        return xr.DataArray(
            expvar,
            dims=['mode'],
            coords={'mode' : self._idx_mode},
            name='explained_variance_ratio'
        )

    def eofs(self, scaling : int = 0) -> List[xr.DataArray]:
        transformers = self._tf
        idx = self._multi_idx_features

        eofs = super().eofs(scaling=scaling)
        eofs = [eofs[idx[i]:idx[i + 1]] for i in range(len(idx) - 1)]
        eofs = [tf.back_transform_eofs(eof) for eof, tf in zip(eofs, transformers)]
        return eofs

    def pcs(self, scaling : int = 0) -> List[xr.DataArray]:
        transformers = self._tf
        idx = self._multi_idx_features

        eofs = super().eofs(scaling=scaling)
        eofs = [eofs[idx[i]:idx[i + 1]] for i in range(len(idx) - 1)]
        eofs = [tf.back_transform_eofs(eof) for eof, tf in zip(eofs, transformers)]

        pcs = super().pcs(scaling=scaling)
        pcs = self._tf.back_transform_pcs(pcs)
        pcs.name = 'PCs'
        return pcs

    def eofs_as_correlation(self) -> List[Tuple[xr.DataArray, xr.DataArray]]:
        # TODO: Implement multivariate
        corr, pvals = super().eofs_as_correlation()
        corr = self._tf.back_transform_eofs(corr)
        pvals = self._tf.back_transform_eofs(pvals)
        corr.name = 'correlation_coeffient'
        pvals.name = 'p_value'
        return corr, pvals

    def reconstruct_X(
        self,
        mode : Optional[Union[int, List[int], slice]] = None
    ) -> xr.DataArray:
        # TODO: multivariate
        Xrec = super().reconstruct_X(mode=mode)
        Xrec = self._tf.back_transform(Xrec)
        coords = {dim: self._tf.coords[dim] for dim in self._tf.dims_samples}
        Xrec = Xrec.assign_coords(coords)
        Xrec.name = 'X_reconstructed'
        return Xrec

    def project_onto_eofs(
        self,
        X : xr.DataArray,
        scaling : int = 0
    ) -> List[xr.DataArray]:
        '''Project new data onto the EOFs.

        Parameters
        ----------
        X : xr.DataArray
             New data to project. Data must have same feature shape as original
             data.
        scaling : [0, 1, 2]
            Projections are scaled (i) to be orthonormal (``scaling=0``), (ii) by the
            square root of the eigenvalues (``scaling=1``) or (iii) by the
            singular values (``scaling=2``). In case no weights were applied,
            scaling by the singular values results in the projections having the
            unit of the input data (the default is 0).

        '''
        # TODO: multivariate
        proj = _DataArrayTransformer()
        X = proj.fit_transform(X, dim=self._tf.dims_samples)
        pcs = super().project_onto_eofs(X=X, scaling=scaling)
        return proj.back_transform_pcs(pcs)
