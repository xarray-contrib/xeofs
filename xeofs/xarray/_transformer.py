from typing import Union, Iterable, List
import numpy as np
import xarray as xr

from ..models._transformer import _ArrayTransformer, _MultiArrayTransformer


class _DataArrayTransformer(_ArrayTransformer):
    '''`_ArrayTransformer` wrapper for `xarray.DataArray`.

    '''

    def __init__(self):
        super().__init__()

    def fit(
        self,
        X : xr.DataArray,
        dim : Union[str, Iterable[str]] = 'time',
        coslat : bool = False
    ):
        if not isinstance(X, xr.DataArray):
            raise ValueError('This interface is for `xarray.DataArray` only')
        # Ensure that dims are tuple
        if isinstance(dim, str):
            dim = [dim]
        dim = tuple(dim)
        # Get axis position for each dim
        if np.isin(dim, X.dims).all():
            axis = [np.argmax(np.isin(X.dims, d)) for d in dim]
        else:
            err_msg = 'One or more of {:} is not a valid dimension.'
            err_msg = err_msg.format(dim)
            raise ValueError(err_msg)

        self.dims = X.dims
        self.dims_samples = dim
        self.dims_features = tuple(d for d in self.dims if d not in dim)

        self.coords = X.coords
        self.coords_samples = {
            k: c for k, c in self.coords.items() if k in self.dims_samples
        }
        self.coords_features = {
            k: c for k, c in self.coords.items() if k in self.dims_features
        }
        if coslat:
            self.coslat_weights = self._get_coslat_weights(X)
        else:
            self.coslat_weights = None

        super().fit(X=X.values, axis=axis)
        return self

    def _get_coslat_weights(self, X : xr.DataArray):
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
        sample_dims = self.dims_samples
        feature_grid = X.isel({k: 0 for k in sample_dims})
        feature_grid = feature_grid.drop_vars(sample_dims)
        return weights.broadcast_like(feature_grid)

    def transform(self, X : xr.DataArray):
        dims_expected = self.dims
        try:
            dims_received = X.dims
        except AttributeError:
            err_msg = 'X must be of type {:}.'.format(repr(xr.DataArray))
            raise TypeError(err_msg)
        if dims_expected == dims_received:
            return super().transform(X.values)
        else:
            err_msg = 'Expected dimensions {:}, but got {:} instead.'
            err_msg = err_msg.format(dims_expected, dims_received)
            raise ValueError(err_msg)

    def fit_transform(
        self,
        X : xr.DataArray,
        dim : Union[str, Iterable[str]] = 'time',
        coslat : bool = False
    ):
        return self.fit(X=X, dim=dim, coslat=coslat).transform(X=X)

    def transform_weights(self, weights : xr.DataArray):
        dims_expected = self.dims_features
        try:
            dims_received = weights.dims
        except AttributeError:
            return super().transform_weights(weights)
        if dims_expected == dims_received:
            return super().transform_weights(weights.values)
        else:
            err_msg = 'Expected dimensions {:}, but got {:} instead.'
            err_msg = err_msg.format(dims_expected, dims_received)
            raise ValueError(err_msg)

    def back_transform(self, X : np.ndarray):
        da = super().back_transform(X)
        return xr.DataArray(
            da, dims=self.dims, coords=self.coords, name='reconstructed_data'
        )

    def back_transform_eofs(self, X : np.ndarray):
        da = super().back_transform_eofs(X)
        dims = self.dims_features + ('mode',)
        coords = self.coords_features
        coords['mode'] = range(1, X.shape[1] + 1)
        return xr.DataArray(da, dims=dims, coords=coords, name='EOFs')

    def back_transform_pcs(self, X : np.ndarray):
        da = super().back_transform_pcs(X)
        dims = self.dims_samples + ('mode',)
        coords = self.coords_samples
        coords['mode'] = range(1, X.shape[1] + 1)
        return xr.DataArray(da, dims=dims, coords=coords, name='PCs')


class _MultiDataArrayTransformer(_MultiArrayTransformer):
    'Transform multiple N-D ``xr.DataArray`` to a single 2D ``np.ndarry``.'
    def __init__(self):
        super().__init__()

    def fit(
        self,
        X : Union[xr.DataArray, List[xr.DataArray]],
        dim : Union[str, Iterable[str]] = 'time',
        coslat : bool = False
    ):
        X = self._convert2list(X)
        self.tfs = [_DataArrayTransformer().fit(x, dim=dim, coslat=coslat) for x in X]

        if len(set([tf.n_valid_samples for tf in self.tfs])) > 1:
            err_msg = 'All individual arrays must have same number of samples.'
            raise ValueError(err_msg)

        self.idx_array_sep = np.cumsum([tf.n_valid_features for tf in self.tfs])
        self.dims_samples = self.tfs[0].dims_samples
        self.coslat_weights = [tf.coslat_weights for tf in self.tfs]
        return self

    def transform(self, X : Union[xr.DataArray, List[xr.DataArray]]) -> np.ndarray:
        return super().transform(X=X)

    def transform_weights(self, weights : Union[xr.DataArray, List[xr.DataArray]]) -> np.ndarray:
        return super().transform_weights(weights=weights)

    def fit_transform(
        self, X : Union[xr.DataArray, List[xr.DataArray]],
        dim : Union[str, Iterable[str]] = 'time',
        coslat : bool = False
    ) -> np.ndarray:
        return self.fit(X=X, dim=dim, coslat=coslat).transform(X)

    def back_transform(self, X : np.ndarray) -> List[xr.DataArray]:
        return super().back_transform(X=X)

    def back_transform_eofs(self, X : np.ndarray) -> List[xr.DataArray]:
        return super().back_transform_eofs(X=X)

    def back_transform_pcs(self, X : np.ndarray) -> List[xr.DataArray]:
        return super().back_transform_pcs(X=X)
