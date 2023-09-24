from typing import Self

import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils.data_types import DataArray


class DataArraySanitizer(BaseEstimator, TransformerMixin):
    """
    Removes NaNs from the feature dimension of a 2D DataArray.

    """

    def __init__(self, sample_name="sample", feature_name="feature"):
        self.sample_name = sample_name
        self.feature_name = feature_name

    def _check_input_type(self, data) -> None:
        if not isinstance(data, xr.DataArray):
            raise ValueError("Input must be an xarray DataArray")

    def _check_input_dims(self, data: DataArray) -> None:
        if set(data.dims) != set([self.sample_name, self.feature_name]):
            raise ValueError(
                "Input must have dimensions ({:}, {:})".format(
                    self.sample_name, self.feature_name
                )
            )

    def _check_input_coords(self, data: DataArray) -> None:
        if not data.coords[self.feature_name].identical(self.feature_coords):
            raise ValueError(
                "Cannot transform data. Feature coordinates are different."
            )

    def fit(self, data: DataArray, y=None) -> Self:
        # Check if input is a DataArray
        self._check_input_type(data)

        # Check if input has the correct dimensions
        self._check_input_dims(data)

        self.feature_coords = data.coords[self.feature_name]

        # Identify NaN locations
        self.is_valid_feature = data.notnull().all(self.sample_name).compute()

        return self

    def transform(self, data: DataArray) -> DataArray:
        # Check if input is a DataArray
        self._check_input_type(data)

        # Check if input has the correct dimensions
        self._check_input_dims(data)

        # Check if input has the correct coordinates
        self._check_input_coords(data)

        # Remove NaN entries
        data = data.isel({self.feature_name: self.is_valid_feature})

        return data

    def fit_transform(self, data: DataArray, y=None) -> DataArray:
        return self.fit(data, y).transform(data)

    def inverse_transform_data(self, data: DataArray) -> DataArray:
        # Reindex only if feature coordinates are different
        is_same_coords = data.coords[self.feature_name].identical(self.feature_coords)

        if is_same_coords:
            return data
        else:
            return data.reindex({self.feature_name: self.feature_coords.values})

    def inverse_transform_components(self, data: DataArray) -> DataArray:
        # Reindex only if feature coordinates are different
        is_same_coords = data.coords[self.feature_name].identical(self.feature_coords)

        if is_same_coords:
            return data
        else:
            return data.reindex({self.feature_name: self.feature_coords.values})

    def inverse_transform_scores(self, data: DataArray) -> DataArray:
        return data
