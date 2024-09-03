import warnings

import numpy as np
from dask.array import Array as DaskArray  # type: ignore
from dask.array.linalg import svd_compressed as dask_svd
from dask.graph_manipulation import wait_on
from scipy.sparse.linalg import svds as complex_svd  # type: ignore
from sklearn.utils.extmath import randomized_svd

from ...utils.sanity_checks import sanity_check_n_modes


def get_deterministic_sign_multiplier(data, axis: int):
    """Compute a sign multiplier that ensures deterministic output using Dask.

    Parameters:
    ------------
    data: da.Array
        Input data to determine sorting order.
    axis: int
        Axis along which to compute the sign multiplier.

    Returns:
    ---------
    sign_multiplier: da.Array
        Sign multiplier that ensures deterministic output.
    """
    max_vals = np.max(data, axis=axis)
    min_vals = np.min(data, axis=axis)
    sign_multiplier = np.where(np.abs(max_vals) >= np.abs(min_vals), 1, -1)
    return sign_multiplier


class _SVD:
    """Decomposes a data object using Singular Value Decomposition (SVD).

    The data object will be decomposed like X = U * S * V.T, where U and V are
    orthogonal matrices and S is a diagonal matrix with the singular values on
    the diagonal.

    Parameters
    ----------
    n_modes : int | float | str
        Number of components to be computed. If float, it represents the fraction of variance to keep. If "all", all components are kept.
    init_rank_reduction : float, default=0.0
        Used only when `n_modes` is given as a float. Specifiy the initial target rank to be computed by randomized SVD before truncating the solution to the desired fraction of explained variance. Must be in the half open interval (0, 1]. Lower values will speed up the computation. If the rank is too low and the fraction of explained variance is not reached, a warning will be raised.
    flip_signs : bool, default=True
        Whether to flip the sign of the components to ensure deterministic output.
    solver : {'auto', 'full', 'randomized'}, default='auto'
        The solver is selected by a default policy based on size of `X` and `n_modes`:
        if the input data is larger than 500x500 and the number of modes to extract is lower
        than 80% of the smallest dimension of the data, then the more efficient
        `randomized` method is enabled. Otherwise the exact full SVD is computed
        and optionally truncated afterwards.
    random_state : np.random.Generator | int | None, default=None
        Seed for the random number generator.
    solver_kwargs : dict, default={}
        Additional keyword arguments passed to the SVD solver.
    """

    def __init__(
        self,
        n_modes: int | float | str,
        init_rank_reduction: float = 0.3,
        flip_signs: bool = True,
        solver: str = "auto",
        random_state: np.random.Generator | int | None = None,
        solver_kwargs: dict = {},
        is_complex: bool | str = "auto",
    ):
        sanity_check_n_modes(n_modes)
        self.is_based_on_variance = True if isinstance(n_modes, float) else False

        if self.is_based_on_variance:
            if not (0 < init_rank_reduction <= 1.0):
                raise ValueError(
                    "init_rank_reduction must be in the half open interval (0, 1]."
                )

        self.n_modes = n_modes
        self.n_modes_precompute = n_modes
        self.init_rank_reduction = init_rank_reduction
        self.flip_signs = flip_signs
        self.solver = solver
        self.random_state = random_state
        self.solver_kwargs = solver_kwargs
        self.is_complex = is_complex

    def _get_n_modes_precompute(self, rank: int) -> int:
        if self.is_based_on_variance:
            n_modes_precompute = int(rank * self.init_rank_reduction)
            if n_modes_precompute < 1:
                warnings.warn(
                    f"`init_rank_reduction={self.init_rank_reduction}` is too low resulting in zero components. One component will be computed instead."
                )
                n_modes_precompute = 1
        elif self.n_modes_precompute == "all":
            n_modes_precompute = rank

        elif isinstance(self.n_modes_precompute, int):
            if self.n_modes_precompute > rank:
                raise ValueError(
                    f"n_modes must be less than or equal to the rank of the dataset (rank = {rank})."
                )
            n_modes_precompute = self.n_modes_precompute
        return n_modes_precompute

    def fit_transform(self, X):
        """Decomposes the data object.

        Parameters
        ----------
        X : DataArray
            A 2-dimensional data object to be decomposed.
        """
        rank = min(X.shape)
        self.n_modes_precompute = self._get_n_modes_precompute(rank)

        # Check if data is small enough to use exact SVD
        # If not, use randomized SVD
        # If data is complex, use scipy sparse SVD
        # If data is dask, use dask SVD
        # Conditions for using exact SVD follow scitkit-learn's PCA implementation
        # Source: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

        use_dask = True if isinstance(X, DaskArray) else False

        match self.is_complex:
            case bool():
                use_complex = self.is_complex
            case "auto":
                use_complex = True if np.iscomplexobj(X) else False
            case _:
                raise ValueError(
                    f"Unrecognized value for is_complex '{self.is_complex}'. "
                    "Valid options are True, False, and 'auto'."
                )

        is_small_data = max(X.shape) < 500

        match self.solver:
            # TODO(nicrie): implement more performant case for tall and skinny problems which are best handled by precomputing the covariance matrix.
            # if X.shape[1] <= 1_000 and X.shape[0] >= 10 * X.shape[1]: -> covariance_eigh" (see sklearn PCA implementation: https://github.com/scikit-learn/scikit-learn/blob/e87b32a81c70abed8f2e97483758eb64df8255e9/sklearn/decomposition/_pca.py#L526)
            case "auto":
                has_many_modes = self.n_modes_precompute > int(0.8 * rank)
                use_exact = (
                    True if is_small_data and has_many_modes and not use_dask else False
                )
            case "full":
                use_exact = True
            case "randomized":
                use_exact = False
            case _:
                raise ValueError(
                    f"Unrecognized solver '{self.solver}'. "
                    "Valid options are 'auto', 'full', and 'randomized'."
                )

        # Use exact SVD for small data sets
        if use_exact:
            U, s, VT = self._svd(X, np.linalg.svd, self.solver_kwargs)
            U = U[:, : self.n_modes_precompute]
            s = s[: self.n_modes_precompute]
            VT = VT[: self.n_modes_precompute, :]

        # Use randomized SVD for large, real-valued data sets
        elif (not use_complex) and (not use_dask):
            solver_kwargs = self.solver_kwargs | {
                "n_components": self.n_modes_precompute,
                "random_state": self.random_state,
            }
            U, s, VT = self._svd(X, randomized_svd, solver_kwargs)

        # Use scipy sparse SVD for large, complex-valued data sets
        elif use_complex and (not use_dask):
            # Scipy sparse version
            solver_kwargs = self.solver_kwargs | {
                "k": self.n_modes_precompute,
                "solver": "lobpcg",
                "random_state": self.random_state,
            }
            U, s, VT = self._svd(X, complex_svd, solver_kwargs)
            idx_sort = np.argsort(s)[::-1]
            U = U[:, idx_sort]
            s = s[idx_sort]
            VT = VT[idx_sort, :]

        # Use dask SVD for large, real-valued, delayed data sets
        elif (not use_complex) and use_dask:
            solver_kwargs = self.solver_kwargs | {
                "k": self.n_modes_precompute,
                "seed": self.random_state,
            }
            solver_kwargs.setdefault("compute", False)
            solver_kwargs.setdefault("n_power_iter", 4)
            U, s, VT = self._svd(X, dask_svd, solver_kwargs)
        else:
            err_msg = (
                "Complex data together with dask is currently not implemented. See dask issue 7639 "
                "https://github.com/dask/dask/issues/7639"
            )
            raise NotImplementedError(err_msg)

        U, s, VT = wait_on(U, s, VT)

        V = VT.conj().T

        # Flip signs of components to ensure deterministic output
        if self.flip_signs:
            sign_multiplier = get_deterministic_sign_multiplier(V, axis=0)
            V *= sign_multiplier
            U *= sign_multiplier

        # Truncate the decomposition to the desired number of modes
        if self.is_based_on_variance:
            # Truncating based on variance requires computation of dask array
            # which we prefer to avoid
            if use_dask:
                err_msg = "Estimating the number of modes to keep based on variance is not supported with dask arrays. Please explicitly specifiy the number of modes to keep by using an integer for the number of modes."
                raise ValueError(err_msg)

            # Compute the fraction of explained variance per mode
            N = X.shape[0] - 1
            total_variance = X.var(axis=0, ddof=1).sum()
            explained_variance = s**2 / N / total_variance
            cum_expvar = explained_variance.cumsum()
            total_explained_variance = cum_expvar[-1].item()

            n_modes_required = (
                self.n_modes_precompute - (cum_expvar >= self.n_modes).sum() + 1
            )
            if n_modes_required > self.n_modes_precompute:
                warnings.warn(
                    f"Dataset has {self.n_modes_precompute} components, explaining {total_explained_variance:.2%} of the variance. However, {self.n_modes:.2%} explained variance was requested. Please consider increasing `init_rank_reduction`."
                )
                n_modes_required = self.n_modes_precompute

            # Truncate solution to the desired fraction of explained variance
            U = U[:, :n_modes_required]
            s = s[:n_modes_required]
            V = V[:, :n_modes_required]

        return U, s, V

    def _svd(self, X, func, kwargs):
        """Performs SVD on the data

        Parameters
        ----------
        X : DataArray
            A 2-dimensional data object to be decomposed.
        func : Callable
            Method to perform SVD.
        kwargs : dict
            Additional keyword arguments passed to the SVD solver.

        Returns
        -------
        U : DataArray
            Left singular vectors.
        s : DataArray
            Singular values.
        VT : DataArray
            Right singular vectors.
        """
        try:
            U, s, VT = func(X, **kwargs)
            return U, s, VT
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError(
                "SVD failed. This may be due to isolated NaN values in the data. Please consider the following steps:\n"
                "1. Check for and remove any isolated NaNs in your dataset.\n"
                "2. If the error persists, please raise an issue at https://github.com/xarray-contrib/xeofs/issues."
            )
