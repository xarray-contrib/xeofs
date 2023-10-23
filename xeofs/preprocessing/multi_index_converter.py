from typing import List, Optional
from typing_extensions import Self
import pandas as pd

from .transformer import Transformer
from ..utils.data_types import Dims, DataArray, DataSet, Data, DataVar, DataVarBound


class MultiIndexConverter(Transformer):
    """Convert MultiIndexes of an ND DataArray or Dataset to regular indexes."""

    def __init__(self):
        super().__init__()
        self.original_indexes = {}
        self.modified_dimensions = []

    def fit(
        self,
        X: Data,
        sample_dims: Optional[Dims] = None,
        feature_dims: Optional[Dims] = None,
        **kwargs
    ) -> Self:
        # Store original MultiIndexes and replace with simple index
        for dim in X.dims:
            index = X.indexes[dim]
            if isinstance(index, pd.MultiIndex):
                self.original_indexes[dim] = X.coords[dim]
                self.modified_dimensions.append(dim)

        return self

    def transform(self, X: DataVar) -> DataVar:
        X_transformed = X.copy(deep=True)

        # Replace MultiIndexes with simple index
        for dim in self.modified_dimensions:
            size = X_transformed.coords[dim].size
            X_transformed = X_transformed.drop_vars(dim)
            X_transformed.coords[dim] = range(size)

        return X_transformed

    def _inverse_transform(self, X: DataVarBound) -> DataVarBound:
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

    def inverse_transform_data(self, X: DataVarBound) -> DataVarBound:
        return self._inverse_transform(X)

    def inverse_transform_components(self, X: DataVarBound) -> DataVarBound:
        return self._inverse_transform(X)

    def inverse_transform_scores(self, X: DataArray) -> DataArray:
        return self._inverse_transform(X)


# class DataListMultiIndexConverter(BaseEstimator, TransformerMixin):
#     """Converts MultiIndexes to simple indexes and vice versa."""

#     def __init__(self):
#         self.converters: List[MultiIndexConverter] = []

#     def fit(self, X: List[Data], y=None):
#         for x in X:
#             converter = MultiIndexConverter()
#             converter.fit(x)
#             self.converters.append(converter)

#         return self

#     def transform(self, X: List[Data]) -> List[Data]:
#         X_transformed: List[Data] = []
#         for x, converter in zip(X, self.converters):
#             X_transformed.append(converter.transform(x))

#         return X_transformed

#     def fit_transform(self, X: List[Data], y=None) -> List[Data]:
#         return self.fit(X, y).transform(X)

#     def _inverse_transform(self, X: List[Data]) -> List[Data]:
#         X_inverse_transformed: List[Data] = []
#         for x, converter in zip(X, self.converters):
#             X_inverse_transformed.append(converter._inverse_transform(x))

#         return X_inverse_transformed

#     def inverse_transform_data(self, X: List[Data]) -> List[Data]:
#         return self._inverse_transform(X)

#     def inverse_transform_components(self, X: List[Data]) -> List[Data]:
#         return self._inverse_transform(X)

#     def inverse_transform_scores(self, X: DataArray) -> DataArray:
#         return self.converters[0].inverse_transform_scores(X)
