from typing import Optional
from typing_extensions import Self
import xarray as xr

from .transformer import Transformer
from ..utils.data_types import Dims, DataArray, DataSet, Data, DataVar


class Sanitizer(Transformer):
    """
    Removes NaNs from the feature dimension of a 2D DataArray.

    """

    def __init__(self, sample_name="sample", feature_name="feature"):
        super().__init__(sample_name=sample_name, feature_name=feature_name)

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
        self.is_valid_feature = ~X.isnull().all(self.sample_name).compute()
        # NOTE: We must also consider the presence of valid samples. For
        # instance, when PCA is applied with "longitude" and "latitude" as
        # sample dimensions, certain grid cells may be masked (e.g., due to
        # ocean areas). To ensure correct reconstruction of scores,
        # we need to identify the sample positions of NaNs in the fitted
        # dataset. Keep in mind that when transforming new data,
        # we have to recheck for valid samples, as the new dataset may have
        # different samples.
        X_valid = X.sel({self.feature_name: self.is_valid_feature})
        self.is_valid_sample = ~X_valid.isnull().all(self.feature_name).compute()

        return self

    def transform(self, X: DataArray) -> DataArray:
        # Check if input is a DataArray
        self._check_input_type(X)

        # Check if input has the correct dimensions
        self._check_input_dims(X)

        # Check if input has the correct coordinates
        self._check_input_coords(X)

        # Remove NaN entries; only consider full-dimensional NaNs
        # We already know valid features from the fitted dataset
        X = X.isel({self.feature_name: self.is_valid_feature})
        # However, we need to recheck for valid samples, as the new dataset may
        # have different samples
        is_valid_sample = ~X.isnull().all(self.feature_name).compute()
        X = X.isel({self.sample_name: is_valid_sample})

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
