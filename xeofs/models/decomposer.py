import numpy as np
import xarray as xr
from dask.array import Array as DaskArray  # type: ignore
from numpy.linalg import svd
from sklearn.utils.extmath import randomized_svd
from scipy.sparse.linalg import svds as complex_svd  # type: ignore
from dask.array.linalg import svd_compressed as dask_svd


class Decomposer:
    """Decomposes a data object using Singular Value Decomposition (SVD).

    The data object will be decomposed like X = U * S * V.T, where U and V are
    orthogonal matrices and S is a diagonal matrix with the singular values on
    the diagonal.

    Parameters
    ----------
    n_modes : int
        Number of components to be computed.
    solver : {'auto', 'full', 'randomized'}, default='auto'
        The solver is selected by a default policy based on size of `X` and `n_modes`:
        if the input data is larger than 500x500 and the number of modes to extract is lower
        than 80% of the smallest dimension of the data, then the more efficient
        `randomized` method is enabled. Otherwise the exact full SVD is computed
        and optionally truncated afterwards.
    **kwargs
        Additional keyword arguments passed to the SVD solver.
    """

    def __init__(self, n_modes=100, flip_signs=True, solver="auto", **kwargs):
        self.n_modes = n_modes
        self.flip_signs = flip_signs
        self.solver = solver
        self.solver_kwargs = kwargs

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

        if self.solver == "auto":
            use_exact = (
                True if is_small_data and self.n_modes > int(0.8 * rank) else False
            )
        elif self.solver == "full":
            use_exact = True
        elif self.solver == "randomized":
            use_exact = False
        else:
            raise ValueError(
                f"Unrecognized solver '{self.solver}'. "
                "Valid options are 'auto', 'full', and 'randomized'."
            )

        # Use exact SVD for small data sets
        if use_exact:
            U, s, VT = xr.apply_ufunc(
                np.linalg.svd,
                X,
                kwargs=self.solver_kwargs,
                input_core_dims=[dims],
                output_core_dims=[
                    [dims[0], "mode"],
                    ["mode"],
                    ["mode", dims[1]],
                ],
                dask="allowed",
                vectorize=False,
            )
            U = U[:, : self.n_modes]
            s = s[: self.n_modes]
            VT = VT[: self.n_modes, :]

        # Use randomized SVD for large, real-valued data sets
        elif (not use_complex) and (not use_dask):
            self.solver_kwargs.update({"n_components": self.n_modes})

            U, s, VT = xr.apply_ufunc(
                randomized_svd,
                X,
                kwargs=self.solver_kwargs,
                input_core_dims=[dims],
                output_core_dims=[
                    [dims[0], "mode"],
                    ["mode"],
                    ["mode", dims[1]],
                ],
            )

        # Use scipy sparse SVD for large, complex-valued data sets
        elif use_complex and (not use_dask):
            # Scipy sparse version
            self.solver_kwargs.update(
                {
                    "k": self.n_modes,
                    "solver": "lobpcg",
                }
            )
            U, s, VT = xr.apply_ufunc(
                complex_svd,
                X,
                kwargs=self.solver_kwargs,
                input_core_dims=[dims],
                output_core_dims=[
                    [dims[0], "mode"],
                    ["mode"],
                    ["mode", dims[1]],
                ],
            )
            idx_sort = np.argsort(s)[::-1]
            U = U[:, idx_sort]
            s = s[idx_sort]
            VT = VT[idx_sort, :]

        # Use dask SVD for large, real-valued, delayed data sets
        elif (not use_complex) and use_dask:
            self.solver_kwargs.update({"k": self.n_modes})
            U, s, VT = xr.apply_ufunc(
                dask_svd,
                X,
                kwargs=self.solver_kwargs,
                input_core_dims=[dims],
                output_core_dims=[
                    [dims[0], "mode"],
                    ["mode"],
                    ["mode", dims[1]],
                ],
                dask="allowed",
            )
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
            # Flip signs of components to ensure deterministic output
            idx_sign = abs(VT).argmax(dims[1]).compute()
            flip_signs = np.sign(VT.isel({dims[1]: idx_sign}))
            flip_signs = flip_signs.compute()
            # Drop all dimensions except 'mode' so that the index is clean
            for dim, coords in flip_signs.coords.items():
                if dim != "mode":
                    flip_signs = flip_signs.drop(dim)
            VT *= flip_signs
            U *= flip_signs

        self.U_ = U
        self.s_ = s
        self.V_ = VT.conj().transpose(dims[1], "mode")
