import xarray as xr

from ._base_scaler import _BaseScaler
from ._base_stacker import _BaseStacker
from .scaler import SingleDataArrayScaler, SingleDatasetScaler, ListDataArrayScaler
from .stacker import SingleDataArrayStacker, SingleDatasetStacker, ListDataArrayStacker
from .multi_index_converter import (
    MultiIndexConverter,
    ListMultiIndexConverter,
)
from ..utils.data_types import AnyDataObject


class ScalerFactory:
    @staticmethod
    def create_scaler(data: AnyDataObject, **kwargs) -> _BaseScaler:
        if isinstance(data, xr.DataArray):
            return SingleDataArrayScaler(**kwargs)
        elif isinstance(data, xr.Dataset):
            return SingleDatasetScaler(**kwargs)
        elif isinstance(data, list) and all(
            isinstance(da, xr.DataArray) for da in data
        ):
            return ListDataArrayScaler(**kwargs)
        else:
            raise ValueError("Invalid data type")


class MultiIndexConverterFactory:
    @staticmethod
    def create_converter(
        data: AnyDataObject, **kwargs
    ) -> MultiIndexConverter | ListMultiIndexConverter:
        if isinstance(data, (xr.DataArray, xr.Dataset)):
            return MultiIndexConverter(**kwargs)
        elif isinstance(data, list) and all(
            isinstance(da, xr.DataArray) for da in data
        ):
            return ListMultiIndexConverter(**kwargs)
        else:
            raise ValueError("Invalid data type")


class StackerFactory:
    @staticmethod
    def create_stacker(data: AnyDataObject, **kwargs) -> _BaseStacker:
        if isinstance(data, xr.DataArray):
            return SingleDataArrayStacker(**kwargs)
        elif isinstance(data, xr.Dataset):
            return SingleDatasetStacker(**kwargs)
        elif isinstance(data, list) and all(
            isinstance(da, xr.DataArray) for da in data
        ):
            return ListDataArrayStacker(**kwargs)
        else:
            raise ValueError("Invalid data type")
