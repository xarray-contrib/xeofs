import xarray as xr

from ..utils.data_types import DataArray
from ._numpy._rotation import _promax


def promax(loadings: DataArray, feature_dim, **kwargs):
    rotated, rot_mat, phi_mat = xr.apply_ufunc(
        _promax,
        loadings,
        input_core_dims=[[feature_dim, "mode"]],
        output_core_dims=[
            [feature_dim, "mode"],
            ["mode_m", "mode_n"],
            ["mode_m", "mode_n"],
        ],
        kwargs=kwargs,
        dask="allowed",
    )

    return rotated, rot_mat, phi_mat
