import xarray as xr

from ._base_stacker import _BaseStacker
from .stacker import SingleDataArrayStacker, SingleDatasetStacker, ListDataArrayStacker
from ..utils.data_types import AnyDataObject

class StackerFactory:
    @staticmethod
    def create_stacker(data: AnyDataObject, **kwargs) -> _BaseStacker:
        if isinstance(data, xr.DataArray):
            return SingleDataArrayStacker(**kwargs)
        elif isinstance(data, xr.Dataset):
            return SingleDatasetStacker(**kwargs)
        elif isinstance(data, list) and all(isinstance(da, xr.DataArray) for da in data):
            return ListDataArrayStacker(**kwargs)
        else:
            raise ValueError("Invalid data type")
