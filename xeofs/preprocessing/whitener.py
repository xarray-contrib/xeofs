import warnings

import numpy as np
import xarray as xr
from typing_extensions import Self

from ..linalg._numpy import _fractional_matrix_power
from ..utils.data_types import (
    DataArray,
    Dims,
    DimsList,
)
from ..utils.sanity_checks import assert_single_dataarray
from .transformer import Transformer


class Whitener(Transformer):
    """Fractional whitening of 2D DataArray.

    For ``alpha=1`` (no whiteneing), it just becomes the identity transformation.

    Parameters
    ----------
    alpha: float, default=0
        Power parameter to perform fractional whitening, where 0 corresponds to full whitening (standardized and decorrelated) and 1 to no whitening.
    sample_name: str, default="sample"
        Name of the sample dimension.
    feature_name: str, default="feature"
        Name of the feature dimension.
    random_state: int | None, default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        alpha: float = 0,
        sample_name: str = "sample",
        feature_name: str = "feature",
        random_state: int | None = None,
    ):
        super().__init__(sample_name, feature_name)

        # Verify that alpha has a lower bound of 0
        if alpha < 0:
            raise ValueError("`alpha` must be greater than or equal to 0")

        alpha = float(alpha)

        self.alpha = alpha
        self.random_state = random_state

        # Check whether Whitener is identity transformation
        self.is_identity = self._check_identity_transform()

    def _check_identity_transform(self) -> bool:
        eps = np.finfo(self.alpha).eps
        alpha_is_one = (1.0 - self.alpha) < eps
        if alpha_is_one:
            return True
        else:
            return False

    def _sanity_check_input(self, X) -> None:
        assert_single_dataarray(X)

        if len(X.dims) != 2:
            raise ValueError("Input DataArray must have shape 2")

        if X.dims != (self.sample_name, self.feature_name):
            raise ValueError(
                "Input DataArray must have dimensions ({:}, {:})".format(
                    self.sample_name, self.feature_name
                )
            )

    def get_serialization_attrs(self) -> dict:
        return dict(
            alpha=self.alpha,
            is_identity=self.is_identity,
            T=self.T,
            Tinv=self.Tinv,
            feature_name=self.feature_name,
            n_samples=self.n_samples,
        )

    def fit(
        self,
        X: DataArray,
        sample_dims: Dims | None = None,
        feature_dims: DimsList | None = None,
    ) -> Self:
        self._sanity_check_input(X)
        n_samples, n_features = X.shape
        self.n_samples = n_samples

        if self.is_identity:
            self.T = xr.DataArray(1, name="identity")
            self.Tinv = xr.DataArray(1, name="identity")

        # Compute the fractional whitening transformation directly based on covariance matrix
        else:
            if n_samples < n_features:
                warnings.warn(
                    f"The number of samples ({n_samples}) is smaller than the number of features ({n_features}), leading to an ill-conditioned problem. This may cause unstable results. Consider using PCA to reduce dimensionality and stabilize the problem by setting `use_pca=True`."
                )

            self.T, self.Tinv = self._compute_whitener_transform(X)

        return self

    def _compute_whitener_transform(self, X: DataArray) -> tuple[DataArray, DataArray]:
        T, Tinv = xr.apply_ufunc(
            self._compute_whitener_transform_numpy,
            X,
            input_core_dims=[[self.sample_name, self.feature_name]],
            output_core_dims=[[self.feature_name, "mode"], ["mode", self.feature_name]],
            dask="allowed",
        )
        T = T.assign_coords(mode=X.coords[self.feature_name].data)
        Tinv = Tinv.assign_coords(mode=X.coords[self.feature_name].data)
        return T, Tinv

    def _compute_whitener_transform_numpy(self, X):
        nc = X.shape[0]
        C = X.conj().T @ X / nc
        power = (self.alpha - 1) / 2
        svd_kwargs = {"random_state": self.random_state, "solver": "full"}
        T = _fractional_matrix_power(C, power, **svd_kwargs)
        try:
            Tinv = np.linalg.inv(T)
        except np.linalg.LinAlgError:
            Tinv = np.linalg.pinv(T)
        return T, Tinv

    def transform(self, X: DataArray) -> DataArray:
        """Transform new data into the fractional whitened space."""

        self._sanity_check_input(X)
        if self.is_identity:
            return X
        else:
            transformed = xr.dot(X, self.T, dims=self.feature_name)
            transformed.name = X.name
            return transformed.rename({"mode": self.feature_name})

    def fit_transform(
        self,
        X: DataArray,
        sample_dims: Dims | None = None,
        feature_dims: DimsList | None = None,
    ) -> DataArray:
        return self.fit(X, sample_dims, feature_dims).transform(X)

    def inverse_transform_data(self, X: DataArray) -> DataArray:
        """Transform 2D data (sample x feature) from whitened space back into original space.

        Parameters
        ----------
        X: DataArray
            Data to transform back into original space.
        """
        if self.is_identity:
            return X
        else:
            X = X.rename({self.feature_name: "mode"})
            return xr.dot(X, self.Tinv, dims="mode")

    def transform_components(self, X: DataArray) -> DataArray:
        """Transform 2D components (feature x mode) into whitened space."""

        if self.is_identity:
            return X
        else:
            dummy_dim = "dummy_dim"
            VS = self.T.conj().T
            VS = VS.rename({"mode": dummy_dim})
            transformed = xr.dot(VS, X, dims=self.feature_name)
            return transformed.rename({dummy_dim: self.feature_name})

    def inverse_transform_components(self, X: DataArray) -> DataArray:
        """Transform 2D components (feature x mode) from whitened space back into original space."""

        if self.is_identity:
            return X
        else:
            dummy_dim = "dummy_dim"
            comps_pc_space = X.rename({self.feature_name: dummy_dim})
            VS = self.Tinv.conj().T
            VS = VS.rename({"mode": dummy_dim})
            return xr.dot(VS, comps_pc_space, dims=dummy_dim)

    def inverse_transform_scores(self, X: DataArray) -> DataArray:
        """Transform 2D scores (sample x mode) from whitened space back into original space."""

        return X

    def inverse_transform_scores_unseen(self, X: DataArray) -> DataArray:
        """Transform unseen 2D scores (sample x mode) from whitened space back into original space."""

        return X
