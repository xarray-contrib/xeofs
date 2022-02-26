from typing import Union, Iterable
import numpy as np
import xarray as xr

from .. import models


class _DataArrayTransformer(models.eof._ArrayTransformer):
    '''`_ArrayTransformer` wrapper for `xarray.DataArray`.

    '''

    def __init__(self):
        super().__init__()

    def fit(self, X : xr.DataArray, dim : Union[str, Iterable[str]] = 'time'):
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

        super().fit(X=X.values, axis=axis)
        return self

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

    def fit_transform(self, X : xr.DataArray, dim : Union[str, Iterable[str]] = 'time'):
        return self.fit(X=X, dim=dim).transform(X=X)

    def transform_weights(self, weights : xr.DataArray):
        if weights is None:
            return None
        dims_expected = self.dims_features
        try:
            dims_received = weights.dims
        except AttributeError:
            err_msg = 'weights must be of type {:}.'.format(repr(xr.DataArray))
            raise TypeError(err_msg)
        if dims_expected == dims_received:
            return super().transform_weights(weights.values)
        else:
            err_msg = 'Expected dimensions {:}, but got {:} instead.'
            err_msg = err_msg.format(dims_expected, dims_received)
            raise ValueError(err_msg)

    def back_transform(self, X : np.ndarray):
        da = super().back_transform(X)
        return xr.DataArray(da, dims=self.dims, coords=self.coords)

    def back_transform_eofs(self, X : np.ndarray):
        da = super().back_transform_eofs(X)
        dims = self.dims_features + ('mode',)
        coords = self.coords_features
        coords['mode'] = range(1, X.shape[1] + 1)
        return xr.DataArray(da, dims=dims, coords=coords)

    def back_transform_pcs(self, X : np.ndarray):
        da = super().back_transform_pcs(X)
        dims = self.dims_samples + ('mode',)
        coords = self.coords_samples
        coords['mode'] = range(1, X.shape[1] + 1)
        return xr.DataArray(da, dims=dims, coords=coords)
