import xarray as xr

from ._base_scaler import _BaseScaler
from .scaler import SingleDataArrayScaler, SingleDatasetScaler, ListDataArrayScaler
from ..utils.data_types import AnyDataObject

class ScalerFactory:
    @staticmethod
    def create_scaler(data: AnyDataObject, **kwargs) -> _BaseScaler:
        if isinstance(data, xr.DataArray):
            return SingleDataArrayScaler(**kwargs)
        elif isinstance(data, xr.Dataset):
            return SingleDatasetScaler(**kwargs)
        elif isinstance(data, list) and all(isinstance(da, xr.DataArray) for da in data):
            return ListDataArrayScaler(**kwargs)
        else:
            raise ValueError("Invalid data type")
