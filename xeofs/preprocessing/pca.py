import numpy as np
import xarray as xr
from typing_extensions import Self

from ..linalg.svd import SVD
from ..utils.data_types import (
    DataArray,
    Dims,
    DimsList,
)
from ..utils.sanity_checks import assert_single_dataarray
from .transformer import Transformer


class PCA(Transformer):
    """Transform data into reduced PC space.

    For large number of features, use PCA to reduce the dimensionality of the data before whitening.

    Note that for ``alpha=1`` (no whiteneing) and ``use_pca=False``, it just becomes the identity transformation.

    Parameters
    ----------
    n_modes: int | float | str, default="all"
        If int, number of components to keep. If float, fraction of variance to keep. If `n_modes="all"`, keep all components.
    use_pca: bool, default=True
        Whether or not to use PCA to reduce the dimensionality of the data. If False, perform identity transformation.
    init_rank_reduction: float, default=0.3
        Used only when `n_modes` is given as a float. Specifiy the initial PCA rank reduction before truncating the solution to the desired fraction of explained variance. Must be in the half open interval ]0, 1]. Lower values will speed up the computation.
    compute_eagerly: bool, default=False
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
        n_modes: int | float | str = "all",
        use_pca: bool = True,
        init_rank_reduction: float = 0.3,
        compute_eagerly: bool = False,
        sample_name: str = "sample",
        feature_name: str = "feature",
        random_state: np.random.Generator | int | None = None,
        solver_kwargs: dict = {},
    ):
        super().__init__(sample_name, feature_name)

        self.use_pca = use_pca
        self.n_modes = n_modes
        self.init_rank_reduction = init_rank_reduction
        self.compute_eagerly = compute_eagerly
        self.random_state = random_state
        self.solver_kwargs = solver_kwargs

        # Check whether Whitener is identity transformation
        self.is_identity = not use_pca

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
            V=self.V,
            use_pca=self.use_pca,
        )

    def fit(
        self,
        X: DataArray,
        sample_dims: Dims | None = None,
        feature_dims: DimsList | None = None,
    ) -> Self:
        self._sanity_check_input(X)

        if self.use_pca:
            # In case of "all" modes to the rank of the input data
            self.n_modes = self._get_n_modes(X)

            svd = SVD(
                n_modes=self.n_modes,
                init_rank_reduction=self.init_rank_reduction,
                compute=self.compute_eagerly,
                random_state=self.random_state,
                sample_name=self.sample_name,
                feature_name=self.feature_name,
                **self.solver_kwargs,
            )
            _, _, self.V = svd.fit_transform(X)

        else:
            self.V = xr.DataArray(1, name="identity")

        return self

    def transform(self, X: DataArray) -> DataArray:
        """Transform new data into the PC space."""

        self._sanity_check_input(X)
        if self.use_pca:
            transformed = xr.dot(X, self.V, dims=self.feature_name)
            transformed.name = X.name
            return transformed.rename({"mode": self.feature_name})
        else:
            return X

    def fit_transform(
        self,
        X: DataArray,
        sample_dims: Dims | None = None,
        feature_dims: DimsList | None = None,
    ) -> DataArray:
        return self.fit(X, sample_dims, feature_dims).transform(X)

    def inverse_transform_data(self, X: DataArray) -> DataArray:
        """Transform 2D data (sample x feature) from PC space back into original space."""
        if self.use_pca:
            X = X.rename({self.feature_name: "mode"})
            return xr.dot(X, self.V.conj().T, dims="mode")
        else:
            return X

    def transform_components(self, X: DataArray) -> DataArray:
        """Transform 2D components (feature x mode) into PC space."""

        if self.use_pca:
            dummy_dim = "dummy_dim"
            Tinv = self.V.conj().T
            Tinv = Tinv.rename({"mode": dummy_dim})
            transformed = xr.dot(Tinv, X, dims=self.feature_name)
            return transformed.rename({dummy_dim: self.feature_name})
        else:
            return X

    def inverse_transform_components(self, X: DataArray) -> DataArray:
        """Transform 2D components (feature x mode) from PC space back into original space."""

        if self.use_pca:
            dummy_dim = "dummy_dim"
            comps_pc_space = X.rename({self.feature_name: dummy_dim})
            V = self.V
            V = V.rename({"mode": dummy_dim})
            return xr.dot(V, comps_pc_space, dims=dummy_dim)
        else:
            return X

    def inverse_transform_scores(self, X: DataArray) -> DataArray:
        """Transform 2D scores (sample x mode) from whitened PC space back into original space."""

        return X

    def inverse_transform_scores_unseen(self, X: DataArray) -> DataArray:
        """Transform unseen 2D scores (sample x mode) from whitened PC space back into original space."""

        return X
