from typing import Dict, Optional

import xarray as xr
from typing_extensions import Self

from ..models.decomposer import Decomposer
from ..utils.data_types import (
    DataArray,
    Dims,
    DimsList,
)
from ..utils.sanity_checks import assert_single_dataarray
from .transformer import Transformer


class Whitener(Transformer):
    """Whiten a 2D DataArray matrix using PCA.

    Parameters
    ----------
    n_modes: int | float
        If int, number of components to keep. If float, fraction of variance to keep.
    init_rank_reduction: float, default=0.3
        Used only when `n_modes` is given as a float. Specifiy the initial PCA rank reduction before truncating the solution to the desired fraction of explained variance. Must be in the half open interval ]0, 1]. Lower values will speed up the computation.
    alpha: float, default=0.0
        Power parameter to perform fractional whitening, where 0 corresponds to full PCA whitening and 1 to PCA without whitening.
    sample_name: str, default="sample"
        Name of the sample dimension.
    feature_name: str, default="feature"
        Name of the feature dimension.
    solver_kwargs: Dict
        Additional keyword arguments for the SVD solver.

    """

    def __init__(
        self,
        n_modes: int | float,
        init_rank_reduction: float = 0.3,
        alpha: float = 0.0,
        sample_name: str = "sample",
        feature_name: str = "feature",
        solver_kwargs: Dict = {},
    ):
        super().__init__(sample_name, feature_name)

        # Verify that alpha has a lower bound of 0
        if alpha < 0:
            raise ValueError("`alpha` must be greater than or equal to 0")

        self.n_modes = n_modes
        self.init_rank_reduction = init_rank_reduction
        self.alpha = alpha
        self.solver_kwargs = solver_kwargs

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

    def get_serialization_attrs(self) -> Dict:
        return dict(n_modes=self.n_modes, alpha=self.alpha)

    def fit(
        self,
        X: xr.DataArray,
        sample_dims: Optional[Dims] = None,
        feature_dims: Optional[DimsList] = None,
    ) -> Self:
        self._sanity_check_input(X)

        decomposer = Decomposer(
            n_modes=self.n_modes,
            init_rank_reduction=self.init_rank_reduction,
            **self.solver_kwargs,
        )
        decomposer.fit(X, dims=(self.sample_name, self.feature_name))

        self.U = decomposer.U_
        self.s = decomposer.s_
        self.V = decomposer.V_

        return self

    def transform(self, X: xr.DataArray) -> DataArray:
        """Transform new data into the fractional whitened PC space."""

        self._sanity_check_input(X)

        scores = xr.dot(X, self.V, dims=self.feature_name) * self.s ** (self.alpha - 1)
        return scores.rename({"mode": self.feature_name})

    def fit_transform(
        self,
        X: xr.DataArray,
        sample_dims: Optional[Dims] = None,
        feature_dims: Optional[DimsList] = None,
    ) -> DataArray:
        return self.fit(X, sample_dims, feature_dims).transform(X)

    def inverse_transform_data(self, X: DataArray) -> DataArray:
        """Transform 2D data (sample x feature) from whitened PC space back into original space."""

        X = X.rename({self.feature_name: "mode"})
        X_unwhitened = X * self.s ** (1 - self.alpha)
        return xr.dot(X_unwhitened, self.V.conj().T, dims="mode")

    def inverse_transform_components(self, X: DataArray) -> DataArray:
        """Transform 2D components (feature x mode) from whitened PC space back into original space."""

        dummy_dim = "dummy_dim"
        comps_pc_space = X.rename({self.feature_name: dummy_dim})
        V = self.V.rename({"mode": dummy_dim})
        return xr.dot(comps_pc_space, V.conj().T, dims=dummy_dim)

    def inverse_transform_scores(self, X: DataArray) -> DataArray:
        """Transform 2D scores (sample x mode) from whitened PC space back into original space."""

        return X

    def inverse_transform_scores_unseen(self, X: DataArray) -> DataArray:
        """Transform unseen 2D scores (sample x mode) from whitened PC space back into original space."""

        return X
