from typing import List, Optional, Dict
from typing_extensions import Self

import pandas as pd
import numpy as np
import xarray as xr

from .transformer import Transformer
from ..utils.data_types import (
    Dims,
    DimsList,
    DataArray,
    DataSet,
    Data,
    DataVar,
    DataList,
    DataArrayList,
    DataSetList,
    DataVarList,
)


class Concatenator(Transformer):
    """Concatenate a list of DataArrays along the feature dimensions."""

    def __init__(self, sample_name: str = "sample", feature_name: str = "feature"):
        super().__init__(sample_name, feature_name)

        self.n_data = None
        self.n_features = []
        self.coords_in = {}

    def get_serialization_attrs(self) -> Dict:
        return dict(
            n_data=self.n_data,
            n_features=self.n_features,
            coords_in=self.coords_in,
        )

    def fit(
        self,
        X: List[DataArray],
        sample_dims: Optional[Dims] = None,
        feature_dims: Optional[DimsList] = None,
    ) -> Self:
        # Check that all inputs are DataArrays
        if not all([isinstance(data, DataArray) for data in X]):
            raise ValueError("Input must be a list of DataArrays")

        # Check that all inputs have shape 2
        if not all([len(data.dims) == 2 for data in X]):
            raise ValueError("Input DataArrays must have shape 2")

        # Check that all inputs have the same sample_name and feature_name
        if not all([data.dims == (self.sample_name, self.feature_name) for data in X]):
            raise ValueError("Input DataArrays must have the same dimensions")

        self.n_data = len(X)

        # Set input feature coordinates, using dict for easier serialization
        self.coords_in = {
            str(i): data.coords[self.feature_name] for i, data in enumerate(X)
        }
        self.n_features = [coord.size for coord in self.coords_in.values()]

        return self

    def transform(self, X: List[DataArray]) -> DataArray:
        # Test whether the input list has same length as the number of stackers
        if len(X) != self.n_data:
            raise ValueError(
                f"Invalid input. Number of DataArrays ({len(X)}) does not match the number of fitted DataArrays ({self.n_data})."
            )

        reindexed_data_list: List[DataArray] = []

        idx_range = np.cumsum([0] + self.n_features)
        for i, data in enumerate(X):
            # Create dummy feature coordinates for DataArray
            new_coords = np.arange(idx_range[i], idx_range[i + 1])

            # Replace original feature coordinates with dummy coordinates
            data = data.drop_vars(self.feature_name)
            reindexed = data.assign_coords({self.feature_name: new_coords})

            reindexed_data_list.append(reindexed)

        X_concat: DataArray = xr.concat(reindexed_data_list, dim=self.feature_name)
        self.coords_out = X_concat.coords[self.feature_name]

        return X_concat

    def fit_transform(
        self,
        X: List[DataArray],
        sample_dims: Optional[Dims] = None,
        feature_dims: Optional[DimsList] = None,
    ) -> DataArray:
        return self.fit(X, sample_dims, feature_dims).transform(X)

    def _split_dataarray_into_list(self, data: DataArray) -> List[DataArray]:
        feature_name = self.feature_name
        data_list: List[DataArray] = []

        idx_range = np.cumsum([0] + self.n_features)
        for i, coords in enumerate(self.coords_in.values()):
            # Create dummy feature coordinates for DataArray
            features = np.arange(idx_range[i], idx_range[i + 1])
            # Select the features corresponding to the current DataArray
            sub_selection = data.sel({feature_name: features})
            # Replace dummy feature coordinates with original feature coordinates
            sub_selection = sub_selection.assign_coords({feature_name: coords})
            data_list.append(sub_selection)

        return data_list

    def inverse_transform_data(self, X: DataArray) -> List[DataArray]:
        """Reshape the 2D data (sample x feature) back into its original shape."""
        return self._split_dataarray_into_list(X)

    def inverse_transform_components(self, X: DataArray) -> List[DataArray]:
        """Reshape the 2D components (sample x feature) back into its original shape."""
        return self._split_dataarray_into_list(X)

    def inverse_transform_scores(self, X: DataArray) -> DataArray:
        """Reshape the 2D scores (sample x mode) back into its original shape."""
        return X

    def inverse_transform_scores_unseen(self, X: DataArray) -> DataArray:
        """Reshape the 2D scores (sample x mode) back into its original shape."""
        return X
