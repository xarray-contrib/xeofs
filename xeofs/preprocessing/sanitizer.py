import xarray as xr
from dask.base import compute
from typing_extensions import Self

from ..utils.data_types import Data, DataArray, Dims
from ..utils.sanity_checks import assert_single_dataarray
from .transformer import Transformer


class Sanitizer(Transformer):
    """
    Removes NaNs from the feature dimension of a 2D DataArray.

    """

    def __init__(self, sample_name="sample", feature_name="feature", check_nans=True):
        super().__init__(sample_name=sample_name, feature_name=feature_name)

        self.check_nans = check_nans

        self.feature_coords = xr.DataArray()
        self.sample_coords = xr.DataArray()
        self.is_valid_feature = xr.DataArray()

    def get_serialization_attrs(self) -> dict:
        return dict(
            feature_coords=self.feature_coords,
            sample_coords=self.sample_coords,
            is_valid_feature=self.is_valid_feature,
        )

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

    def _get_valid_features_per_sample(self, X: Data) -> Data:
        """Isolated NaN check that an be constructed lazily, where we check that
        the number of valid features is the same for every sample, with
        the exception of all-NaN samples."""
        return X.notnull().sum(self.feature_name)

    def fit(
        self,
        X: Data,
        sample_dims: Dims | None = None,
        feature_dims: Dims | None = None,
        **kwargs,
    ) -> Self:
        # Check if input is a DataArray
        assert_single_dataarray(X)

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
        assert_single_dataarray(X)

        # Check if input has the correct dimensions
        self._check_input_dims(X)

        # Check if input has the correct coordinates
        self._check_input_coords(X)

        X_valid_features = self._get_valid_features(X)
        X_valid_samples = self._get_valid_samples(X)
        X_valid_features_per_sample = self._get_valid_features_per_sample(X)

        # Optionally skip NaN checks to preserve lazy computation for dask arrays
        if self.check_nans:
            (
                self.is_valid_feature,
                X_valid_features,
                X_valid_samples,
                X_valid_features_per_sample,
            ) = compute(
                self.is_valid_feature,
                X_valid_features,
                X_valid_samples,
                X_valid_features_per_sample,
            )

            # Validate that non-NaN features match the original from .fit()
            if not X_valid_features.equals(self.is_valid_feature):
                raise ValueError(
                    "Input data had NaN features in different locations than"
                    " the original data."
                )

            isolated_nans = ~X_valid_features_per_sample.isin(
                [0, X_valid_features.sum().values]
            )
            if isolated_nans.any():
                raise ValueError(
                    "Input data contains partial NaN entries, which will cause the"
                    " the SVD to fail."
                )

            X = X.where(X_valid_features & X_valid_samples, drop=True)

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
