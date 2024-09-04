import numpy as np
import xarray as xr
from dask.base import compute as dask_compute

from ..utils.data_types import DataArray
from ._numpy import _SVD


class SVD:
    def __init__(
        self,
        n_modes: int | float | str,
        is_complex: bool | str = "auto",
        init_rank_reduction: float = 0.3,
        flip_signs: bool = True,
        solver: str = "auto",
        compute: bool = True,
        random_state: np.random.Generator | int | None = None,
        solver_kwargs: dict = {},
        sample_name: str = "sample",
        feature_name: str = "feature",
    ):
        self.n_modes = n_modes
        self.is_complex = is_complex
        self.init_rank_reduction = init_rank_reduction
        self.flip_signs = flip_signs
        self.solver = solver
        self.random_state = random_state
        self.solver_kwargs = solver_kwargs
        self.compute_svd = compute

        self.sample_name = sample_name
        self.feature_name = feature_name

    def fit_transform(self, X: DataArray) -> tuple[DataArray, DataArray, DataArray]:
        """Decomposes the data object.

        Parameters
        ----------
        X : DataArray
            A 2-dimensional data object to be decomposed.

        Returns
        -------
        U : DataArray
            The left singular vectors of the decomposition.
        s : DataArray
            The singular values of the decomposition.
        V : DataArray
            The right singular vectors of the decomposition.

        """
        svd = _SVD(
            n_modes=self.n_modes,
            init_rank_reduction=self.init_rank_reduction,
            flip_signs=self.flip_signs,
            solver=self.solver,
            random_state=self.random_state,
            is_complex=self.is_complex,
            **self.solver_kwargs,
        )
        U, s, V = xr.apply_ufunc(
            svd.fit_transform,
            X,
            input_core_dims=[[self.sample_name, self.feature_name]],
            output_core_dims=[
                [self.sample_name, "mode"],
                ["mode"],
                [self.feature_name, "mode"],
            ],
            dask="allowed",
        )
        mode_coords = np.arange(1, U.mode.size + 1)
        s = s.assign_coords(mode=mode_coords)
        U = U.assign_coords(mode=mode_coords)
        V = V.assign_coords(mode=mode_coords)

        if self.compute_svd:
            U, s, V = dask_compute(U, s, V)

        return U, s, V
