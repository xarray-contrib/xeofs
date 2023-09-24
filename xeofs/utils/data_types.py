from typing import (
    List,
    TypeAlias,
    Sequence,
    TypedDict,
    Optional,
    Tuple,
    TypeVar,
    Hashable,
)

import xarray as xr
import dask.array as da

DataArray: TypeAlias = xr.DataArray
DataSet: TypeAlias = xr.Dataset
DataList: TypeAlias = List[xr.DataArray]
DaskArray: TypeAlias = da.Array  # type: ignore
DataObject: TypeAlias = DataArray | DataSet | DataList
DataX2 = TypeVar("DataX2", DataArray, DataSet)
DataX3 = TypeVar("DataX3", DataArray, DataSet, DataList)


Dims: TypeAlias = Sequence[Hashable]
DimsTuple: TypeAlias = Tuple[Dims, ...]
DimsList: TypeAlias = List[Dims]
DimsListTuple: TypeAlias = Tuple[DimsList, ...]


# Replace this with the above
Dataset: TypeAlias = xr.Dataset
DataArrayList: TypeAlias = List[DataArray]
SingleDataObject = TypeVar("SingleDataObject", DataArray, Dataset)
XArrayData = TypeVar("XArrayData", DataArray, Dataset)
AnyDataObject = TypeVar("AnyDataObject", DataArray, Dataset, DataArrayList)
