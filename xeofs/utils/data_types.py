from typing import Hashable, Sequence, TypeAlias, TypeVar

import dask.array as da
from xarray.core import dataarray as xr_dataarray
from xarray.core import dataset as xr_dataset

DataArray: TypeAlias = xr_dataarray.DataArray
DataSet: TypeAlias = xr_dataset.Dataset
Data: TypeAlias = DataArray | DataSet
DataVar = TypeVar("DataVar", DataArray, DataSet)
DataVarBound = TypeVar("DataVarBound", bound=Data)

DataArrayList: TypeAlias = list[DataArray]
DataSetList: TypeAlias = list[DataSet]
DataList: TypeAlias = list[Data]
DataVarList: TypeAlias = list[DataVar]

GenericType = TypeVar("GenericType")

DaskArray: TypeAlias = da.Array  # type: ignore
DataObject: TypeAlias = DataArray | DataSet | DataList

Dims: TypeAlias = Sequence[Hashable]
DimsTuple: TypeAlias = tuple[Dims, ...]
DimsList: TypeAlias = list[Dims]
DimsListTuple: TypeAlias = tuple[DimsList, ...]
