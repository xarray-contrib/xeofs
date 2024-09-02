import warnings

import numpy as np
import xarray as xr
from typing_extensions import Self

from ..linalg._numpy import _fractional_matrix_power
from ..linalg.svd import SVD
from ..utils.data_types import (
    DataArray,
    Dims,
    DimsList,
)
from ..utils.sanity_checks import assert_single_dataarray
from .transformer import Transformer


class Whitener(Transformer):
    """Fractional whitening of 2D DataArray.

    For large number of features, use PCA to reduce the dimensionality of the data before whitening.

    Note that for ``alpha=1`` (no whiteneing) and ``use_pca=False``, it just becomes the identity transformation.

    Parameters
    ----------
    alpha: float, default=0
        Power parameter to perform fractional whitening, where 0 corresponds to full whitening (standardized and decorrelated) and 1 to no whitening.
    use_pca: bool, default=False
        If True, perform PCA before whitening to speed up the computation. This is the recommended setting for large number of features. Specify the number of components to keep in `n_modes`.
    n_modes: int | float | str, default=None
        If int, number of components to keep. If float, fraction of variance to keep. If `n_modes="all"`, keep all components.
    init_rank_reduction: float, default=0.3
        Used only when `n_modes` is given as a float. Specifiy the initial PCA rank reduction before truncating the solution to the desired fraction of explained variance. Must be in the half open interval ]0, 1]. Lower values will speed up the computation.
    compute_svd: bool, default=False
        Whether to perform eager or lazy computation.
    sample_name: str, default="sample"
        Name of the sample dimension.
    feature_name: str, default="feature"
        Name of the feature dimension.
    random_state: np.random.Generator | int | None, default=None
        Random seed for reproducibility.
    solver_kwargs: dict
        Additional keyword arguments for the SVD solver.

    """

    def __init__(
        self,
        alpha: float = 0.0,
        use_pca: bool = False,
        n_modes: int | float | str = "all",
        init_rank_reduction: float = 0.3,
        compute_svd: bool = False,
        sample_name: str = "sample",
        feature_name: str = "feature",
        random_state: np.random.Generator | int | None = None,
        solver_kwargs: dict = {},
    ):
        super().__init__(sample_name, feature_name)

        # Verify that alpha has a lower bound of 0
        if alpha < 0:
            raise ValueError("`alpha` must be greater than or equal to 0")

        alpha = float(alpha)

        self.alpha = alpha
        self.use_pca = use_pca
        self.n_modes = n_modes
        self.init_rank_reduction = init_rank_reduction
        self.compute_svd = compute_svd
        self.random_state = random_state
        self.solver_kwargs = solver_kwargs

        # Check whether Whitener is identity transformation
        self.is_identity = self._check_identity_transform()

    def _check_identity_transform(self) -> bool:
        eps = np.finfo(self.alpha).eps
        alpha_is_one = (1.0 - self.alpha) < eps
        if not self.use_pca and alpha_is_one:
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

    def _get_n_modes(self, X: DataArray) -> int | float:
        if isinstance(self.n_modes, str):
            if self.n_modes == "all":
                return min(X.shape)
            else:
                raise ValueError("`n_modes` must be an integer, float or 'all'")
        else:
            return self.n_modes

    def get_serialization_attrs(self) -> dict:
        return dict(
            alpha=self.alpha,
            n_modes=self.n_modes,
            use_pca=self.use_pca,
            is_identity=self.is_identity,
            s=self.s,
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
            self.s = xr.DataArray(1, name="identity")

        else:
            if self.use_pca:
                # In case of "all" modes to the rank of the input data
                self.n_modes = self._get_n_modes(X)

                svd = SVD(
                    n_modes=self.n_modes,
                    init_rank_reduction=self.init_rank_reduction,
                    compute=self.compute_svd,
                    random_state=self.random_state,
                    sample_name=self.sample_name,
                    feature_name=self.feature_name,
                    **self.solver_kwargs,
                )
                _, s, V = svd.fit_transform(X)

                n_c: float = np.sqrt(n_samples - 1)
                self.T: DataArray = V * (s / n_c) ** (self.alpha - 1)
                self.Tinv = (s / n_c) ** (1 - self.alpha) * V.conj().T
                self.s = s

            # Without PCA compute the fractional whitening transformation directly based on covariance matrix
            else:
                if n_samples < n_features:
                    warnings.warn(
                        f"The number of samples ({n_samples}) is smaller than the number of features ({n_features}), leading to an ill-conditioned problem. This may cause unstable results. Consider using PCA to reduce dimensionality and stabilize the problem by setting `use_pca=True`."
                    )

                self.T, self.Tinv = self._compute_whitener_transform(X)
                self.s = xr.DataArray(1, name="identity")

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
        nc = X.shape[0] - 1
        C = X.conj().T @ X / nc
        power = (self.alpha - 1) / 2
        svd_kwargs = {"random_state": self.random_state}
        T = _fractional_matrix_power(C, power, **svd_kwargs)
        Tinv = np.linalg.inv(T)
        return T, Tinv

    def get_Tinv(self, unwhiten_only=False) -> DataArray:
        """Get the inverse transformation to unwhiten the data without PC transform.

        In contrast to `inverse_transform()`, this method returns the inverse transformation matrix without the PC transformation. That is, for PC transormed data this transformation only unwhitens the data without transforming back into the input space. For non-PC transformed data, this transformation is equivalent to the inverse transformation.
        """
        if self.use_pca and unwhiten_only:
            n_c = np.sqrt(self.n_samples - 1)
            Tinv = (self.s / n_c) ** (1 - self.alpha)
            Tinv = xr.apply_ufunc(
                np.diag,
                Tinv,
                input_core_dims=[["mode"]],
                output_core_dims=[["mode", self.feature_name]],
                dask="allowed",
            )
            Tinv = Tinv.assign_coords({self.feature_name: self.s.coords["mode"].data})
            return Tinv
        else:
            return self.Tinv

    def transform(self, X: DataArray) -> DataArray:
        """Transform new data into the fractional whitened PC space."""

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

    def inverse_transform_data(self, X: DataArray, unwhiten_only=False) -> DataArray:
        """Transform 2D data (sample x feature) from whitened PC space back into original space.

        Parameters
        ----------
        X: DataArray
            Data to transform back into original space.
        unwhiten_only: bool, default=False
            If True, only unwhiten the data without transforming back into the input space. This is useful when the data was transformed with PCA before whitening and you need the unwhitened data in the PC space.
        """
        T_inv = self.get_Tinv(unwhiten_only=unwhiten_only)
        if self.is_identity:
            return X
        else:
            X = X.rename({self.feature_name: "mode"})
            return xr.dot(X, T_inv, dims="mode")

    def transform_components(self, X: DataArray) -> DataArray:
        """Transform 2D components (feature x mode) into whitened PC space."""

        if self.is_identity:
            return X
        else:
            dummy_dim = "dummy_dim"
            VS = self.T.conj().T
            VS = VS.rename({"mode": dummy_dim})
            transformed = xr.dot(VS, X, dims=self.feature_name)
            return transformed.rename({dummy_dim: self.feature_name})

    def inverse_transform_components(self, X: DataArray) -> DataArray:
        """Transform 2D components (feature x mode) from whitened PC space back into original space."""

        if self.is_identity:
            return X
        else:
            dummy_dim = "dummy_dim"
            comps_pc_space = X.rename({self.feature_name: dummy_dim})
            VS = self.Tinv.conj().T
            VS = VS.rename({"mode": dummy_dim})
            return xr.dot(VS, comps_pc_space, dims=dummy_dim)

    def inverse_transform_scores(self, X: DataArray) -> DataArray:
        """Transform 2D scores (sample x mode) from whitened PC space back into original space."""

        return X

    def inverse_transform_scores_unseen(self, X: DataArray) -> DataArray:
        """Transform unseen 2D scores (sample x mode) from whitened PC space back into original space."""

        return X
