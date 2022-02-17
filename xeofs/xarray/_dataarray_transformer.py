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
        # Convert dim to list
        if isinstance(dim, str):
            dim = list([dim])
        # Get axis position for each dim
        if np.isin(dim, X.dims).all():
            axis = [np.argmax(np.isin(X.dims, d)) for d in dim]
        else:
            err_msg = 'One or more of {:} is not a valid dimension.'
            err_msg = err_msg.format(dim)
            raise ValueError(err_msg)

        self.dims = X.dims
        self.dims_samples = dim
        self.dims_features = [d for d in self.dims if d not in dim]

        self.coords = X.coords

        super().fit(X=X.values, axis=axis)
        return self

    def transform(self, X : xr.DataArray):
        expected_dims = np.array(self.dims)
        provided_dims = np.array(X.dims)
        if np.isin(expected_dims, provided_dims).all():
            return super().transform(X.values)
        else:
            err_msg = 'Expected dimensions {:}, but got {:} instead.'
            err_msg = err_msg.format(expected_dims)
            raise ValueError(err_msg)

    def fit_transform(self, X : xr.DataArray, dim : Union[str, Iterable[str]] = 'time'):
        return self.fit(X=X, dim=dim).transform(X=X)

    def back_transform(self, X : np.ndarray):
        da = super().back_transform(X)
        return xr.DataArray(da, dims=self.dims)

    def back_transform_eofs(self, X : np.ndarray):
        da = super().back_transform_eofs(X)
        dims = self.dims_features + ['mode']
        coords = {k: c for k, c in self.coords.items() if k in self.dims_features}
        coords['mode'] = range(1, X.shape[1] + 1)
        return xr.DataArray(da, dims=dims, coords=coords)

    def back_transform_pcs(self, X : np.ndarray):
        da = super().back_transform_pcs(X)
        dims = self.dims_samples + ['mode']
        coords = {k: c for k, c in self.coords.items() if k in self.dims_samples}
        coords['mode'] = range(1, X.shape[1] + 1)
        return xr.DataArray(da, dims=dims, coords=coords)
