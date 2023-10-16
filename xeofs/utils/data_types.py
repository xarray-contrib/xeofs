from typing import (
    List,
    TypeAlias,
    Sequence,
    Tuple,
    TypeVar,
    Hashable,
)

import dask.array as da
import xarray as xr
from xarray.core import dataarray as xr_dataarray
from xarray.core import dataset as xr_dataset

DataArray: TypeAlias = xr_dataarray.DataArray
DataSet: TypeAlias = xr_dataset.Dataset
Data: TypeAlias = DataArray | DataSet
DataVar = TypeVar("DataVar", DataArray, DataSet)
DataVarBound = TypeVar("DataVarBound", bound=Data)

DataArrayList: TypeAlias = List[DataArray]
DataSetList: TypeAlias = List[DataSet]
DataList: TypeAlias = List[Data]
DataVarList: TypeAlias = List[DataVar]


DaskArray: TypeAlias = da.Array  # type: ignore
DataObject: TypeAlias = DataArray | DataSet | DataList

Dims: TypeAlias = Sequence[Hashable]
DimsTuple: TypeAlias = Tuple[Dims, ...]
DimsList: TypeAlias = List[Dims]
DimsListTuple: TypeAlias = Tuple[DimsList, ...]
