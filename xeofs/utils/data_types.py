from typing import List, TypeAlias, TypedDict, Optional, Tuple, TypeVar

import xarray as xr

DataArray: TypeAlias = xr.DataArray
Dataset: TypeAlias = xr.Dataset
DataArrayList: TypeAlias = List[DataArray]
SingleDataObject = TypeVar('SingleDataObject', DataArray, Dataset)
AnyDataObject = TypeVar('AnyDataObject', DataArray, Dataset, DataArrayList)

XarrayData: TypeAlias = DataArray | Dataset
# Model dimensions are always 2-dimensional: sample and feature
Dims: TypeAlias = Tuple[str]
DimsList: TypeAlias = List[Dims]
SampleDims: TypeAlias = Dims
FeatureDims: TypeAlias = Dims | DimsList
# can be either like ('lat', 'lon') (1 DataArray) or (('lat', 'lon'), ('lon')) (multiple DataArrays)
ModelDims = TypedDict('ModelDims', {'sample': SampleDims, 'feature': FeatureDims})
