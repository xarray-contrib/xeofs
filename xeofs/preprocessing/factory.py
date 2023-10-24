# import xarray as xr

# from .scaler import DataArrayScaler, DataSetScaler, DataListScaler
# from .stacker import DataArrayStacker, DataSetStacker, DataListStacker
# from .multi_index_converter import (
#     DataArrayMultiIndexConverter,
#     DataSetMultiIndexConverter,
#     DataListMultiIndexConverter,
# )
# from ..utils.data_types import DataObject


# class ScalerFactory:
#     @staticmethod
#     def create_scaler(data: DataObject, **kwargs):
#         if isinstance(data, xr.DataArray):
#             return DataArrayScaler(**kwargs)
#         elif isinstance(data, xr.Dataset):
#             return DataSetScaler(**kwargs)
#         elif isinstance(data, list) and all(
#             isinstance(da, xr.DataArray) for da in data
#         ):
#             return DataListScaler(**kwargs)
#         else:
#             raise ValueError("Invalid data type")


# class MultiIndexConverterFactory:
#     @staticmethod
#     def create_converter(
#         data: DataObject, **kwargs
#     ) -> DataArrayMultiIndexConverter | DataListMultiIndexConverter:
#         if isinstance(data, xr.DataArray):
#             return DataArrayMultiIndexConverter(**kwargs)
#         elif isinstance(data, xr.Dataset):
#             return DataSetMultiIndexConverter(**kwargs)
#         elif isinstance(data, list) and all(
#             isinstance(da, xr.DataArray) for da in data
#         ):
#             return DataListMultiIndexConverter(**kwargs)
#         else:
#             raise ValueError("Invalid data type")


# class StackerFactory:
#     @staticmethod
#     def create_stacker(data: DataObject, **kwargs):
#         if isinstance(data, xr.DataArray):
#             return DataArrayStacker(**kwargs)
#         elif isinstance(data, xr.Dataset):
#             return DataSetStacker(**kwargs)
#         elif isinstance(data, list) and all(
#             isinstance(da, xr.DataArray) for da in data
#         ):
#             return DataListStacker(**kwargs)
#         else:
#             raise ValueError("Invalid data type")
