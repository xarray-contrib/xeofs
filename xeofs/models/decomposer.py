import numpy as np
import xarray as xr
import dask
from dask.array import Array as DaskArray  # type: ignore
from dask.diagnostics.progress import ProgressBar
from numpy.linalg import svd
from sklearn.utils.extmath import randomized_svd
from scipy.sparse.linalg import svds as complex_svd  # type: ignore
from dask.array.linalg import svd_compressed as dask_svd
from typing import Optional

from ..utils.xarray_utils import get_deterministic_sign_multiplier


class Decomposer:
    """Decomposes a data object using Singular Value Decomposition (SVD).

    The data object will be decomposed like X = U * S * V.T, where U and V are
    orthogonal matrices and S is a diagonal matrix with the singular values on
    the diagonal.

    Parameters
    ----------
    n_modes : int
        Number of components to be computed.
    flip_signs : bool, default=True
        Whether to flip the sign of the components to ensure deterministic output.
    compute : bool, default=True
        Whether to compute the decomposition immediately.
    solver : {'auto', 'full', 'randomized'}, default='auto'
        The solver is selected by a default policy based on size of `X` and `n_modes`:
        if the input data is larger than 500x500 and the number of modes to extract is lower
        than 80% of the smallest dimension of the data, then the more efficient
        `randomized` method is enabled. Otherwise the exact full SVD is computed
        and optionally truncated afterwards.
    random_state : Optional[int], default=None
        Seed for the random number generator.
    verbose: bool, default=False
        Whether to show a progress bar when computing the decomposition.
    solver_kwargs : dict, default={}
        Additional keyword arguments passed to the SVD solver.
    """

    def __init__(
        self,
        n_modes: int,
        flip_signs: bool = True,
        compute: bool = True,
        solver: str = "auto",
        random_state: Optional[int] = None,
        verbose: bool = False,
        solver_kwargs: dict = {},
    ):
        self.n_modes = n_modes
        self.flip_signs = flip_signs
        self.compute = compute
        self.verbose = verbose
        self.solver = solver
        self.random_state = random_state
        self.solver_kwargs = solver_kwargs

    def fit(self, X, dims=("sample", "feature")):
        """Decomposes the data object.

        Parameters
        ----------
        X : DataArray
            A 2-dimensional data object to be decomposed.
        dims : tuple of str
            Dimensions of the data object.
        """
        n_coords1 = len(X.coords[dims[0]])
        n_coords2 = len(X.coords[dims[1]])
        rank = min(n_coords1, n_coords2)

        if self.n_modes > rank:
            raise ValueError(
                f"n_modes must be smaller or equal to the rank of the data object (rank={rank})"
            )

        # Check if data is small enough to use exact SVD
        # If not, use randomized SVD
        # If data is complex, use scipy sparse SVD
        # If data is dask, use dask SVD
        # Conditions for using exact SVD follow scitkit-learn's PCA implementation
        # Source: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

        use_dask = True if isinstance(X.data, DaskArray) else False
        use_complex = True if np.iscomplexobj(X.data) else False

        is_small_data = max(n_coords1, n_coords2) < 500

        match self.solver:
            case "auto":
                use_exact = (
                    True if is_small_data and self.n_modes > int(0.8 * rank) else False
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
            U, s, VT = self._svd(X, dims, np.linalg.svd, self.solver_kwargs)
            U = U[:, : self.n_modes]
            s = s[: self.n_modes]
            VT = VT[: self.n_modes, :]

        # Use randomized SVD for large, real-valued data sets
        elif (not use_complex) and (not use_dask):
            solver_kwargs = self.solver_kwargs | {
                "n_components": self.n_modes,
                "random_state": self.random_state,
            }
            U, s, VT = self._svd(X, dims, randomized_svd, solver_kwargs)

        # Use scipy sparse SVD for large, complex-valued data sets
        elif use_complex and (not use_dask):
            # Scipy sparse version
            solver_kwargs = self.solver_kwargs | {
                "k": self.n_modes,
                "solver": "lobpcg",
                "random_state": self.random_state,
            }
            U, s, VT = self._svd(X, dims, complex_svd, solver_kwargs)
            idx_sort = np.argsort(s)[::-1]
            U = U[:, idx_sort]
            s = s[idx_sort]
            VT = VT[idx_sort, :]

        # Use dask SVD for large, real-valued, delayed data sets
        elif (not use_complex) and use_dask:
            solver_kwargs = self.solver_kwargs | {
                "k": self.n_modes,
                "seed": self.random_state,
            }
            solver_kwargs.setdefault("compute", self.compute)
            solver_kwargs.setdefault("n_power_iter", 4)
            U, s, VT = self._svd(X, dims, dask_svd, solver_kwargs)
            U, s, VT = self._compute_svd_result(U, s, VT)
        else:
            err_msg = (
                "Complex data together with dask is currently not implemented. See dask issue 7639 "
                "https://github.com/dask/dask/issues/7639"
            )
            raise NotImplementedError(err_msg)

        U = U.assign_coords(mode=range(1, U.mode.size + 1))
        s = s.assign_coords(mode=range(1, U.mode.size + 1))
        VT = VT.assign_coords(mode=range(1, U.mode.size + 1))

        U.name = "U"
        s.name = "s"
        VT.name = "V"

        if self.flip_signs:
            sign_multiplier = get_deterministic_sign_multiplier(VT, dims[1])
            VT *= sign_multiplier
            U *= sign_multiplier

        self.U_ = U
        self.s_ = s
        self.V_ = VT.conj().transpose(dims[1], "mode")

    def _svd(self, X, dims, func, kwargs):
        """Performs SVD on the data

        Parameters
        ----------
        X : DataArray
            A 2-dimensional data object to be decomposed.
        dims : tuple of str
            Dimensions of the data object.
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
            U, s, VT = xr.apply_ufunc(
                func,
                X,
                kwargs=kwargs,
                input_core_dims=[dims],
                output_core_dims=[
                    [dims[0], "mode"],
                    ["mode"],
                    ["mode", dims[1]],
                ],
                dask="allowed",
            )
            return U, s, VT
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError(
                "SVD failed. This may be due to isolated NaN values in the data. Please consider the following steps:\n"
                "1. Check for and remove any isolated NaNs in your dataset.\n"
                "2. If the error persists, please raise an issue at https://github.com/xarray-contrib/xeofs/issues."
            )

    def _compute_svd_result(self, U, s, VT):
        """Computes the SVD result.

        Parameters
        ----------
        U : DataArray
            Left singular vectors.
        s : DataArray
            Singular values.
        VT : DataArray
            Right singular vectors.

        Returns
        -------
        U : DataArray
            Left singular vectors.
        s : DataArray
            Singular values.
        VT : DataArray
            Right singular vectors.
        """
        match self.compute:
            case False:
                pass
            case True:
                match self.verbose:
                    case True:
                        with ProgressBar():
                            U, s, VT = dask.compute(U, s, VT)
                    case False:
                        U, s, VT = dask.compute(U, s, VT)
        return U, s, VT
