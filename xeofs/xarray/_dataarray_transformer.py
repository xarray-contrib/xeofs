import numpy as np
import xarray as xr

from .. import models


class _DataArrayTransformer(models.eof._ArrayTransformer):
    '''`_ArrayTransformer` wrapper for `xarray.DataArray`.

    '''

    def __init__(self):
        super().__init__()

    def fit(self, data : xr.DataArray, dim : str = 'time'):
        self.coords_in = data.coords
        self.dims_sample = tuple([dim])
        self.dims_feature = tuple(d for d in data.dims if d not in self.dims_sample)
        self.dims_in = data.dims
        self.dims_out = self.dims_sample + self.dims_feature
        data = data.transpose(*self.dims_out)
        super().fit(data.values)
        return self

    def transform(self, data : xr.DataArray):
        expected_dims = np.array(self.dims_out)
        provided_dims = np.array(data.dims)
        if np.isin(expected_dims, provided_dims).all():
            data = data.transpose(*self.dims_out)
        else:
            raise ValueError('Input dimension must be {:}'.format(expected_dims))

        return super().transform(data.values)

    def fit_transform(self, data : xr.DataArray, dim : str = 'time'):
        return self.fit(data, dim=dim).transform(data)

    def back_transform(self, data : np.ndarray):
        da = super().back_transform(data)
        coords = {d: self.coords_in[d] for d in self.dims_feature}
        da = xr.DataArray(da, dims=self.dims_out, coords=coords)
        return da.transpose(*self.dims_in)
