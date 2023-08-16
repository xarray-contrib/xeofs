from typing import Dict, TypeVar, List


import xarray as xr
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils.data_types import DataArray, XArrayData, AnyDataObject, DataArrayList


class MultiIndexConverter(BaseEstimator, TransformerMixin):
    def __init__(self, return_copy=False):
        self.original_indexes = {}
        self.modified_dimensions = []
        self.return_copy = return_copy

    def fit(self, X: XArrayData, y=None):
        # Check if input is a DataArray or Dataset
        if not isinstance(X, (xr.DataArray, xr.Dataset)):
            raise ValueError("Input must be an xarray DataArray or Dataset")

        # Store original MultiIndexes and replace with simple index
        for dim in X.dims:
            index = X.indexes[dim]
            if isinstance(index, pd.MultiIndex):
                self.original_indexes[dim] = X.coords[dim]
                self.modified_dimensions.append(dim)

        return self

    def transform(self, X: XArrayData) -> XArrayData:
        # Check if input is a DataArray or Dataset
        if not isinstance(X, (xr.DataArray, xr.Dataset)):
            raise ValueError("Input must be an xarray DataArray or Dataset")

        # Make a copy if return_copy is True
        X_transformed = X.copy(deep=True) if self.return_copy else X

        # Replace MultiIndexes with simple index
        for dim in self.modified_dimensions:
            size = X_transformed.coords[dim].size
            X_transformed = X_transformed.drop_vars(dim)
            X_transformed.coords[dim] = range(size)

        return X_transformed

    def inverse_transform(self, X: XArrayData) -> XArrayData:
        # Check if input is a DataArray or Dataset
        if not isinstance(X, (xr.DataArray, xr.Dataset)):
            raise ValueError("Input must be an xarray DataArray or Dataset")

        # Make a copy if return_copy is True
        X_inverse_transformed = X.copy() if self.return_copy else X

        # Restore original MultiIndexes
        for dim, original_index in self.original_indexes.items():
            if dim in X_inverse_transformed.dims:
                X_inverse_transformed.coords[dim] = original_index
                # Set indexes to original MultiIndexes
                indexes = [
                    idx
                    for idx in self.original_indexes[dim].indexes.keys()
                    if idx != dim
                ]
                X_inverse_transformed = X_inverse_transformed.set_index({dim: indexes})

        return X_inverse_transformed

    def fit_transform(self, X: XArrayData, y=None) -> XArrayData:
        return self.fit(X, y).transform(X)


class ListMultiIndexConverter(BaseEstimator, TransformerMixin):
    def __init__(self, return_copy=False):
        self.converters: List[MultiIndexConverter] = []
        self.return_copy = return_copy

    def fit(self, X: DataArrayList, y=None):
        # Check if input is a List of DataArrays
        if not isinstance(X, list) or not all(isinstance(x, xr.DataArray) for x in X):
            raise ValueError("Input must be a list of xarray DataArray")

        for x in X:
            converter = MultiIndexConverter(return_copy=self.return_copy)
            converter.fit(x)
            self.converters.append(converter)

        return self

    def transform(self, X: DataArrayList) -> DataArrayList:
        # Check if input is a List of DataArrays
        if not isinstance(X, list) or not all(isinstance(x, xr.DataArray) for x in X):
            raise ValueError("Input must be a list of xarray DataArray")

        X_transformed: List[DataArray] = []
        for x, converter in zip(X, self.converters):
            X_transformed.append(converter.transform(x))

        return X_transformed

    def inverse_transform(self, X: DataArrayList) -> DataArrayList | DataArray:
        # Data & components are stored in a list of DataArrays
        if isinstance(X, list):
            X_inverse_transformed: List[DataArray] = []
            for x, converter in zip(X, self.converters):
                X_inverse_transformed.append(converter.inverse_transform(x))

            return X_inverse_transformed
        # Scores are stored as a DataArray
        else:
            return self.converters[0].inverse_transform(X)

    def fit_transform(self, X: DataArrayList, y=None) -> DataArrayList:
        return self.fit(X, y).transform(X)
