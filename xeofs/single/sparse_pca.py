# %%

import numpy as np
import xarray as xr
from typing_extensions import Self

from ..utils.data_types import DataArray, DataObject
from ..utils.sanity_checks import assert_not_complex
from ..utils.xarray_utils import get_matrix_rank
from ..utils.xarray_utils import total_variance as compute_total_variance
from ._numpy._sparse_pca import compute_rspca, compute_spca
from .base_model_single_set import BaseModelSingleSet


class SparsePCA(BaseModelSingleSet):
    """
    Sparse PCA via Variable Projection.

    Given a data matrix, Sparse PCA via Variable Projection [1]_ computes a set of sparse components that can optimally reconstruct the input data. The amount of sparsity is controlled by the L1 penalty coefficient, specified by the parameter `alpha`. Additionally, ridge shrinkage can be applied to improve conditioning.

    This implementation uses randomized methods for linear algebra to accelerate computations. The quality of the approximation can be controlled via the oversampling parameter `oversample` and `n_subspace`, which specifies the number of subspace iterations.

    Variable Projection for Sparse PCA solves the following optimization problem:

        minimize :math:`\\frac{1}{2} \\| X - X B A^T \\|^2 + \\alpha \\|B\\|_1 + \\frac{1}{2} \\beta \\|B\\|^2`

    Parameters
    ----------
    n_modes : int, default=2
        Number of modes to calculate.
    alpha : float, default=0.001
        Sparsity controlling parameter. Higher values lead to sparser components.
    beta : float, default=0.001
        Amount of ridge shrinkage to apply in order to improve conditioning.
    regularizer : {'l0', 'l1'}, default='l1'
        Type of sparsity-inducing regularizer. The L1 norm (also known as LASSO)
        leads to a soft-threshold operator (default). The L0 norm is implemented
        via a hard-threshold operator.
    max_iter : int, default=500
        Maximum number of iterations to perform before exiting.
    tol : float, default=1e-5
        Stopping tolerance for reconstruction error.
    robust : bool, default=False
        Use a robust algorithm to compute the sparse PCA.
    oversample : int, default=10
        Controls the oversampling of column space. Increasing this parameter
        may improve numerical accuracy.
    n_subspace : int, default=2
        Parameter to control the number of subspace iterations. Increasing this
        parameter may improve numerical accuracy.
    n_blocks : int, default=2
        Parameter to control how many blocks of columns the input matrix
        should be split into. A larger number requires less fast memory, but
        increases computational time.
    standardize : bool, default=False
        Whether to standardize the input data, i.e., each feature will have a variance of 1.
    use_coslat : bool, default=False
        Whether to use cosine of latitude for scaling.
    sample_name : str, default="sample"
        Name of the sample dimension.
    feature_name : str, default="feature"
        Name of the feature dimension.
    compute : bool, default=True
        Whether to compute elements of the model eagerly, or to defer computation.
        If True, the following steps will be computed sequentially:
        1) the preprocessor scaler,
        2) optional NaN checks,
        3) SVD decomposition,
        4) scores and components.
    random_state : int, optional
        Seed for the random number generator.
    solver : {"auto", "full", "randomized"}, default="randomized"
        Solver to use for the SVD computation.
    solver_kwargs : dict, default={}
        Additional keyword arguments to be passed to the SVD solver.

    References
    ----------
    .. [1] Erichson, N. B. et al. Sparse Principal Component Analysis via Variable Projection. SIAM J. Appl. Math. 80, 977-1002 (2020).

    Notes
    -----
    This implementation is adapted from the code provided by the authors of the paper [1]_, which is part of the Ristretto library (https://github.com/erichson/ristretto).

    Examples
    --------
    >>> model = xe.single.SparsePCA(n_modes=2, alpha=1e-4)
    >>> model.fit(data, "time")
    >>> components = model.components()
    """

    def __init__(
        self,
        n_modes: int = 2,
        alpha: float = 1e-3,
        beta: float = 1e-3,
        robust: bool = False,
        regularizer: str = "l1",
        max_iter: int = 500,
        tol: float = 1e-6,
        oversample: int = 10,
        n_subspace: int = 1,
        n_blocks: int = 1,
        center: bool = True,
        standardize: bool = False,
        use_coslat: bool = False,
        sample_name: str = "sample",
        feature_name: str = "feature",
        check_nans=True,
        compute: bool = True,
        random_state: int | None = None,
        solver: str = "auto",
        solver_kwargs: dict = {},
        **kwargs,
    ):
        super().__init__(
            n_modes=n_modes,
            center=center,
            standardize=standardize,
            use_coslat=use_coslat,
            check_nans=check_nans,
            sample_name=sample_name,
            feature_name=feature_name,
            compute=compute,
            random_state=random_state,
            solver=solver,
            solver_kwargs=solver_kwargs,
            **kwargs,
        )
        self.attrs.update({"model": "Sparse PCA"})
        self._params.update(
            {
                "alpha": alpha,
                "beta": beta,
                "robust": robust,
                "regularizer": regularizer,
                "max_iter": max_iter,
                "tol": tol,
                "oversample": oversample,
                "n_subspace": n_subspace,
                "n_blocks": n_blocks,
            }
        )

    def _fit_algorithm(self, X: DataArray) -> Self:
        sample_name = self.sample_name
        feature_name = self.feature_name

        # Check if the data is real
        # NOTE: Complex data is not supported, it's likely possible but current numpy implementation
        # of sparse_pca needs to be adpated, mainly changing matrix transpose to conjugate transpose.
        # http://arxiv.org/abs/1804.00341
        assert_not_complex(X)

        # Compute the total variance
        total_variance = compute_total_variance(X, dim=sample_name)

        # Compute matrix rank
        rank = get_matrix_rank(X)

        # Decide whether to use exact or randomized algorithm
        is_small_data = max(X.shape) < 500
        solver = self._params["solver"]

        match solver:
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
                    f"Unrecognized solver '{solver}'. "
                    "Valid options are 'auto', 'full', and 'randomized'."
                )

        decomposing_kwargs = dict(
            n_components=self.n_modes,
            alpha=self._params["alpha"],
            beta=self._params["beta"],
            robust=self._params["robust"],
            regularizer=self._params["regularizer"],
            max_iter=self._params["max_iter"],
            tol=self._params["tol"],
            compute=self._params["compute"],
        )
        if use_exact:
            decomposing_algorithm = compute_spca
        else:
            decomposing_algorithm = compute_rspca
            decomposing_kwargs.update(
                dict(
                    oversample=self._params["oversample"],
                    n_subspace=self._params["n_subspace"],
                    n_blocks=self._params["n_blocks"],
                    random_state=self._params["random_state"],
                )
            )

        # Fit the data
        # We obtain the follwing outputs defined in Erichson et al. 2020
        # variable : notation used by Erichson et al.
        # components : sparse weight matrix B
        # components_normal : orthonormal matrix A
        # exp_var : eigenvalues
        components, components_normal, exp_var = xr.apply_ufunc(
            decomposing_algorithm,
            X,
            input_core_dims=[[sample_name, feature_name]],
            output_core_dims=[[feature_name, "mode"], [feature_name, "mode"], ["mode"]],
            dask="allowed",
            kwargs=decomposing_kwargs,
        )

        # Add coordinates to the results
        exp_var.name = "explained_variance"
        exp_var = exp_var.assign_coords({"mode": np.arange(1, self.n_modes + 1)})

        components.name = "sparse_weight_vectors"
        components = components.assign_coords(
            {
                feature_name: X.coords[feature_name],
                "mode": np.arange(1, self.n_modes + 1),
            },
        )

        components_normal.name = "orthonormal_weight_vectors"
        components_normal = components_normal.assign_coords(
            {
                feature_name: X.coords[feature_name],
                "mode": np.arange(1, self.n_modes + 1),
            },
        )

        # Transform the data
        scores = xr.dot(X, components, dims=feature_name)
        scores.name = "scores"

        norms = xr.apply_ufunc(
            np.linalg.norm,
            scores,
            input_core_dims=[[sample_name]],
            output_core_dims=[[]],
            exclude_dims=set((sample_name,)),
            kwargs={"axis": -1},
            dask="allowed",
            vectorize=False,
        )
        norms.name = "component_norms"

        # Store the results
        self.data.add(X, "input_data", allow_compute=False)
        self.data.add(components, "components")
        self.data.add(components_normal, "components_normal")
        self.data.add(scores, "scores")
        self.data.add(norms, "norms")
        self.data.add(exp_var, "explained_variance")
        self.data.add(total_variance, "total_variance")

        self.data.set_attrs(self.attrs)
        return self

    def _transform_algorithm(self, X: DataObject) -> DataArray:
        feature_name = self.preprocessor.feature_name

        components = self.data["components"]

        # Project the data
        projections = xr.dot(X, components, dims=feature_name)
        projections.name = "scores"

        return projections

    def _inverse_transform_algorithm(self, scores: DataArray) -> DataArray:
        # Reconstruct the data
        comps = self.data["components_normal"].sel(mode=scores.mode)

        reconstructed_data = xr.dot(comps.conj(), scores, dims="mode")
        reconstructed_data.name = "reconstructed_data"

        return reconstructed_data

    def components(self) -> DataObject:
        """Return the sparse components.

        The components are given by the sparse weight matrix B in the optimization problem.

        Returns
        -------
        components: DataArray | Dataset | list[DataArray]
            Components of the fitted model.

        """
        return super().components()

    def scores(self, normalized: bool = False) -> DataArray:
        """Return the component scores.

        The component scores :math:`U` are defined as the projection of the fitted
        data :math:`X` onto the sparse weight components :math:`B`.

        .. math::
            U = X B


        Parameters
        ----------
        normalized : bool, default=False
            Whether to normalize the scores by the L2 norm.

        Returns
        -------
        components: DataArray | Dataset | list[DataArray]
            Scores of the fitted model.

        """
        return super().scores(normalized=normalized)

    def explained_variance(self) -> DataArray:
        """Return the explained variance.


        Returns
        -------
        explained_variance: DataArray
            Explained variance.
        """
        return self.data["explained_variance"]

    def explained_variance_ratio(self) -> DataArray:
        """Return explained variance ratio.

        The explained variance ratio :math:`\\gamma_i` is the variance explained
        by each mode normalized by the total variance. It is defined as

        .. math::
            \\gamma_i = \\frac{\\lambda_i}{\\sum_{j=1}^M \\lambda_j}

        where :math:`\\lambda_i` is the explained variance of the :math:`i`-th mode and :math:`M` is the total number of modes.

        Returns
        -------
        explained_variance_ratio: DataArray
            Explained variance ratio.
        """
        exp_var_ratio = self.data["explained_variance"] / self.data["total_variance"]
        exp_var_ratio.attrs.update(self.data["explained_variance"].attrs)
        exp_var_ratio.name = "explained_variance_ratio"
        return exp_var_ratio
