from typing import Optional, Dict
from typing_extensions import Self
import xarray as xr

from .transformer import Transformer
from ..utils.data_types import Dims, DataArray, DataSet, Data, DataVar
from ..utils.xarray_utils import data_is_dask


class Sanitizer(Transformer):
    """
    Removes NaNs from the feature dimension of a 2D DataArray.

    """

    def __init__(self, sample_name="sample", feature_name="feature"):
        super().__init__(sample_name=sample_name, feature_name=feature_name)

        # Set a flag so that we don't sanitize the feature dimension on first run
        self.has_run = False

        self.feature_coords = xr.DataArray()
        self.sample_coords = xr.DataArray()
        self.is_valid_feature = xr.DataArray()

    def get_serialization_attrs(self) -> Dict:
        return dict(
            feature_coords=self.feature_coords,
            sample_coords=self.sample_coords,
            is_valid_feature=self.is_valid_feature,
        )

    def _check_input_type(self, X) -> None:
        if not isinstance(X, xr.DataArray):
            raise ValueError("Input must be an xarray DataArray")

    def _check_input_dims(self, X) -> None:
        if set(X.dims) != set([self.sample_name, self.feature_name]):
            raise ValueError(
                "Input must have dimensions ({:}, {:})".format(
                    self.sample_name, self.feature_name
                )
            )

    def _check_input_coords(self, X) -> None:
        if not X.coords[self.feature_name].identical(self.feature_coords):
            raise ValueError(
                "Cannot transform data. Feature coordinates are different."
            )

    def _get_valid_features(self, X: Data) -> Data:
        return X.notnull().any(self.sample_name)

    def _get_valid_samples(self, X: Data) -> Data:
        return X.notnull().any(self.feature_name)

    def fit(
        self,
        X: Data,
        sample_dims: Optional[Dims] = None,
        feature_dims: Optional[Dims] = None,
        **kwargs
    ) -> Self:
        # Check if input is a DataArray
        self._check_input_type(X)

        # Check if input has the correct dimensions
        self._check_input_dims(X)

        self.feature_coords = X.coords[self.feature_name]
        self.sample_coords = X.coords[self.sample_name]

        # Identify NaN locations
        self.is_valid_feature = self._get_valid_features(X)
        self.is_valid_sample = self._get_valid_samples(X)

        return self

    def transform(self, X: DataArray) -> DataArray:
        # Check if input is a DataArray
        self._check_input_type(X)

        # Check if input has the correct dimensions
        self._check_input_dims(X)

        # Check if input has the correct coordinates
        self._check_input_coords(X)

        X_valid_features = self._get_valid_features(X)

        # Remove full-dimensional NaN entries
        X = X.dropna(dim=self.sample_name, how="all")
        X = X.dropna(dim=self.feature_name, how="all")

        # For new data only, validate that NaN features match the original
        if self.has_run:
            if not X_valid_features.equals(self.is_valid_feature):
                raise ValueError(
                    "Input data had NaN features in different locations than"
                    " than the original data."
                )

        # Only carry out isolated NaN check for non-dask-backed data
        if not data_is_dask(X):
            if X.isnull().any():
                raise ValueError(
                    "Input data contains partial NaN entries, which will cause the"
                    " the SVD to fail."
                )

        # On future runs, run the feature NaN check
        self.has_run = True

        return X

    def inverse_transform_data(self, X: DataArray) -> DataArray:
        # Reindex only if feature coordinates are different
        coords_are_equal = X.coords[self.feature_name].identical(self.feature_coords)

        if coords_are_equal:
            return X
        else:
            return X.reindex({self.feature_name: self.feature_coords.values})

    def inverse_transform_components(self, X: DataArray) -> DataArray:
        # Reindex only if feature coordinates are different
        coords_are_equal = X.coords[self.feature_name].identical(self.feature_coords)

        if coords_are_equal:
            return X
        else:
            return X.reindex({self.feature_name: self.feature_coords.values})

    def inverse_transform_scores(self, X: DataArray) -> DataArray:
        # Reindex only if sample coordinates are different
        coords_are_equal = X.coords[self.sample_name].identical(self.sample_coords)

        if coords_are_equal:
            return X
        else:
            return X.reindex({self.sample_name: self.sample_coords.values})

    def inverse_transform_scores_unseen(self, X: DataArray) -> DataArray:
        # Don't check sample coords for unseen data
        return X
