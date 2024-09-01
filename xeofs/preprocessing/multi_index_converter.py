import pandas as pd
from typing_extensions import Self

from ..utils.data_types import Data, DataArray, DataVar, DataVarBound, Dims
from .transformer import Transformer


class MultiIndexConverter(Transformer):
    """Convert MultiIndexes of an ND DataArray or Dataset to regular indexes."""

    def __init__(self):
        super().__init__()
        self.modified_dimensions = []
        self.coords_from_fit = {}
        self.coords_from_transform = {}

    def get_serialization_attrs(self) -> dict:
        return dict(
            modified_dimensions=self.modified_dimensions,
            coords_from_fit=self.coords_from_fit,
            coords_from_transform=self.coords_from_transform,
        )

    def fit(
        self,
        X: Data,
        sample_dims: Dims | None = None,
        feature_dims: Dims | None = None,
        **kwargs,
    ) -> Self:
        # Store original MultiIndexes
        for dim in X.dims:
            index = X.indexes[dim]
            if isinstance(index, pd.MultiIndex):
                self.coords_from_fit[dim] = X.coords[dim]
                self.modified_dimensions.append(dim)

        return self

    def transform(self, X: DataVar) -> DataVar:
        X_transformed = X.copy(deep=True)

        # Replace MultiIndexes with simple index
        for dim in self.modified_dimensions:
            # We need to store the indexes from "unseen" data
            self.coords_from_transform[dim] = X_transformed.coords[dim]

            index = X_transformed.indexes[dim]
            X_transformed = X_transformed.drop_vars(dim)
            X_transformed.coords[dim] = range(index.size)

        return X_transformed

    def _inverse_transform(self, X: DataVarBound, reference: str) -> DataVarBound:
        X_inverse_transformed = X.copy(deep=True)

        match reference:
            case "fit":
                reference_indexes = self.coords_from_fit
            case "transform":
                reference_indexes = self.coords_from_transform

        # Restore original MultiIndexes
        for dim, original_index in reference_indexes.items():
            if dim in X_inverse_transformed.dims:
                X_inverse_transformed.coords[dim] = original_index
                # Set indexes to original MultiIndexes
                indexes = [idx for idx in original_index.indexes.keys() if idx != dim]
                X_inverse_transformed = X_inverse_transformed.set_index({dim: indexes})

        return X_inverse_transformed

    def inverse_transform_data(self, X: DataVarBound) -> DataVarBound:
        return self._inverse_transform(X, reference="fit")

    def inverse_transform_components(self, X: DataVarBound) -> DataVarBound:
        return self._inverse_transform(X, reference="fit")

    def inverse_transform_scores(self, X: DataArray) -> DataArray:
        return self._inverse_transform(X, reference="fit")

    def inverse_transform_scores_unseen(self, X: DataArray) -> DataArray:
        return self._inverse_transform(X, reference="transform")
