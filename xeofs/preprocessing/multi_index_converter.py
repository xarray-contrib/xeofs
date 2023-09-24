from typing import List, Self

import xarray as xr
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from xeofs.utils.data_types import DataArray

from ..utils.data_types import DataArray, DataSet, DataList


class DataArrayMultiIndexConverter(BaseEstimator, TransformerMixin):
    """Convert MultiIndexes of a ND DataArray to regular indexes."""

    def __init__(self):
        self.original_indexes = {}
        self.modified_dimensions = []

    def fit(self, X: DataArray, y=None) -> Self:
        # Store original MultiIndexes and replace with simple index
        for dim in X.dims:
            index = X.indexes[dim]
            if isinstance(index, pd.MultiIndex):
                self.original_indexes[dim] = X.coords[dim]
                self.modified_dimensions.append(dim)

        return self

    def transform(self, X: DataArray) -> DataArray:
        X_transformed = X.copy(deep=True)

        # Replace MultiIndexes with simple index
        for dim in self.modified_dimensions:
            size = X_transformed.coords[dim].size
            X_transformed = X_transformed.drop_vars(dim)
            X_transformed.coords[dim] = range(size)

        return X_transformed

    def fit_transform(self, X: DataArray, y=None) -> DataArray:
        return self.fit(X, y).transform(X)

    def _inverse_transform(self, X: DataArray) -> DataArray:
        X_inverse_transformed = X.copy(deep=True)

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

    def inverse_transform_data(self, X: DataArray) -> DataArray:
        return self._inverse_transform(X)

    def inverse_transform_components(self, X: DataArray) -> DataArray:
        return self._inverse_transform(X)

    def inverse_transform_scores(self, X: DataArray) -> DataArray:
        return self._inverse_transform(X)


class DataSetMultiIndexConverter(DataArrayMultiIndexConverter):
    """Converts MultiIndexes to simple indexes and vice versa."""

    def fit(self, X: DataSet, y=None) -> Self:
        return super().fit(X, y)  # type: ignore

    def transform(self, X: DataSet) -> DataSet:
        return super().transform(X)  # type: ignore

    def fit_transform(self, X: DataSet, y=None) -> DataSet:
        return super().fit_transform(X, y)  # type: ignore

    def inverse_transform_data(self, X: DataSet) -> DataSet:
        return super().inverse_transform_data(X)  # type: ignore

    def inverse_transform_components(self, X: DataSet) -> DataSet:
        return super().inverse_transform_components(X)  # type: ignore


class DataListMultiIndexConverter(BaseEstimator, TransformerMixin):
    """Converts MultiIndexes to simple indexes and vice versa."""

    def __init__(self):
        self.converters: List[DataArrayMultiIndexConverter] = []

    def fit(self, X: DataList, y=None):
        for x in X:
            converter = DataArrayMultiIndexConverter()
            converter.fit(x)
            self.converters.append(converter)

        return self

    def transform(self, X: DataList) -> DataList:
        X_transformed: List[DataArray] = []
        for x, converter in zip(X, self.converters):
            X_transformed.append(converter.transform(x))

        return X_transformed

    def fit_transform(self, X: DataList, y=None) -> DataList:
        return self.fit(X, y).transform(X)

    def _inverse_transform(self, X: DataList) -> DataList:
        X_inverse_transformed: List[DataArray] = []
        for x, converter in zip(X, self.converters):
            X_inverse_transformed.append(converter._inverse_transform(x))

        return X_inverse_transformed

    def inverse_transform_data(self, X: DataList) -> DataList:
        return self._inverse_transform(X)

    def inverse_transform_components(self, X: DataList) -> DataList:
        return self._inverse_transform(X)

    def inverse_transform_scores(self, X: DataArray) -> DataArray:
        return self.converters[0].inverse_transform_scores(X)
