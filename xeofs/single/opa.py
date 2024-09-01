import numpy as np
import xarray as xr
from typing_extensions import Self

from ..linalg.decomposer import Decomposer
from ..utils.data_types import DataArray, DataObject
from ..utils.sanity_checks import assert_not_complex
from .base_model_single_set import BaseModelSingleSet
from .eof import EOF


class OPA(BaseModelSingleSet):
    """Optimal Persistence Analysis.

    Optimal Persistence Analysis (OPA) [1]_ [2]_ identifies the patterns with the
    largest decorrelation time in a time-varying field, known as optimal
    persistence patterns or optimally persistent patterns (OPP).

    Parameters
    ----------
    n_modes : int
        Number of optimal persistence patterns (OPP) to be computed.
    tau_max : int
        Maximum time lag for the computation of the covariance matrix.
    center : bool, default=True
        Whether to center the input data.
    standardize : bool, default=False
        Whether to standardize the input data.
    use_coslat : bool, default=False
        Whether to use cosine of latitude for scaling.
    n_pca_modes : int
        Number of modes to be computed in the pre-processing step using EOF.
    compute : bool, default=True
        Whether to compute elements of the model eagerly, or to defer computation.
        If True, four pieces of the fit will be computed sequentially: 1) the
        preprocessor scaler, 2) optional NaN checks, 3) SVD decomposition, 4) scores
        and components.
    sample_name : str, default="sample"
        Name of the sample dimension.
    feature_name : str, default="feature"
        Name of the feature dimension.
    solver : {"auto", "full", "randomized"}, default="auto"
        Solver to use for the SVD computation.
    solver_kwargs : dict, default={}
        Additional keyword arguments to pass to the solver.

    References
    ----------
    .. [1] DelSole, T. Optimally Persistent Patterns in Time-Varying Fields. Journal of the Atmospheric Sciences 58, 1341–1356 (2001).
    .. [2] DelSole, T. Low-Frequency Variations of Surface Temperature in Observations and Simulations. Journal of Climate 19, 4487–4507 (2006).

    Examples
    --------
    >>> from xeofs.single import OPA
    >>> model = OPA(n_modes=10, tau_max=50, n_pca_modes=100)
    >>> model.fit(X, dim=("time"))

    Retrieve the optimally persistent patterns (OPP) and their time series:

    >>> opp = model.components()
    >>> opp_ts = model.scores()

    Retrieve the decorrelation time of the OPPs:

    >>> decorrelation_time = model.decorrelation_time()
    """

    def __init__(
        self,
        n_modes: int,
        tau_max: int,
        center: bool = True,
        standardize: bool = False,
        use_coslat: bool = False,
        check_nans: bool = True,
        n_pca_modes: int = 100,
        compute: bool = True,
        sample_name: str = "sample",
        feature_name: str = "feature",
        solver: str = "auto",
        random_state: int | None = None,
        solver_kwargs: dict = {},
    ):
        if n_modes > n_pca_modes:
            raise ValueError(
                f"n_modes must be smaller or equal to n_pca_modes (n_modes={n_modes}, n_pca_modes={n_pca_modes})"
            )
        super().__init__(
            n_modes=n_modes,
            center=center,
            standardize=standardize,
            use_coslat=use_coslat,
            check_nans=check_nans,
            compute=compute,
            sample_name=sample_name,
            feature_name=feature_name,
            solver=solver,
            random_state=random_state,
            solver_kwargs=solver_kwargs,
        )
        self.attrs.update({"model": "OPA"})
        self._params.update({"tau_max": tau_max, "n_pca_modes": n_pca_modes})

    def _Ctau(self, X, tau: int) -> DataArray:
        """Compute the time-lage covariance matrix C(tau) of the data X."""
        sample_name = self.preprocessor.sample_name
        X0 = X.copy(deep=True)
        Xtau = X.shift({sample_name: -tau}).dropna(sample_name)

        X0 = X0.rename({"mode": "feature1"})
        Xtau = Xtau.rename({"mode": "feature2"})

        n_samples = Xtau[sample_name].size
        return xr.dot(X0, Xtau, dims=[sample_name]) / (n_samples - 1)

    @staticmethod
    def _compute_matrix_inverse(X, dims):
        """Compute the inverse of a symmetric matrix X."""
        return xr.apply_ufunc(
            np.linalg.inv,
            X,
            input_core_dims=[dims],
            output_core_dims=[dims[::-1]],
            vectorize=False,
            dask="allowed",
        )

    def _fit_algorithm(self, X: DataArray) -> Self:
        assert_not_complex(X)

        sample_name = self.sample_name
        feature_name = self.feature_name

        # Perform PCA as a pre-processing step
        pca = EOF(
            n_modes=self._params["n_pca_modes"],
            standardize=False,
            use_coslat=False,
            sample_name=self.sample_name,
            feature_name=self.feature_name,
            solver=self._params["solver"],
            compute=self._params["compute"],
            random_state=self._params["random_state"],
            check_nans=False,
            solver_kwargs=self._params["solver_kwargs"],
        )
        pca.fit(X, dim=sample_name)
        n_samples = X.coords[sample_name].size
        comps = pca.data["components"] * np.sqrt(n_samples - 1)
        # -> comps (feature x mode)
        scores = pca.data["scores"] / np.sqrt(n_samples - 1)
        # -> scores (sample x mode)

        # Compute the covariance matrix with zero time lag
        C0 = self._Ctau(scores, 0)
        # -> C0 (feature1 x feature2)
        # C0inv = self._compute_matrix_inverse(C0, dims=("feature1", "feature2"))
        # -> C0inv (feature2 x feature1)
        M = 0.5 * C0
        # -> M (feature1 x feature2)
        tau_max = self._params["tau_max"]
        for tau in range(1, tau_max + 1):
            Ctau = self._Ctau(scores, tau)
            if tau == tau_max:
                Ctau = 0.5 * Ctau
            M = M + (Ctau)

        MT = xr.DataArray(M.data.T, dims=M.dims, coords=M.coords)
        # -> MT (feature1 x feature2)
        M_summed = M + MT
        # -> M_summed (feature1 x feature2)

        # Instead of solving the generalized eigenvalue problem
        # as proposed in DelSole (2001), we solve the
        # eigenvalue problem of the alternativ formulation
        # using a symmtric matrix given in
        # A. Hannachi (2021), Patterns Identification and
        # Data Mining in Weather and Climate, Equation (8.20)
        decomposer = Decomposer(
            n_modes=C0.shape[0],
            flip_signs=False,
            compute=self._params["compute"],
            solver="full",
        )
        decomposer.fit(C0, dims=("feature1", "feature2"))
        C0_sqrt = decomposer.U_ * np.sqrt(decomposer.s_)
        # -> C0_sqrt (feature1 x mode)
        C0_sqrt_inv = self._compute_matrix_inverse(C0_sqrt, dims=("feature1", "mode"))
        # -> C0_sqrt_inv (mode x feature1)
        target = 0.5 * xr.dot(C0_sqrt_inv, M_summed, dims="feature1")
        # -> target (mode x feature2)
        target = xr.dot(
            target, C0_sqrt_inv.rename({"mode": "feature2"}), dims="feature2"
        )
        # -> target (mode x feature1)
        target = target.rename({"feature1": "dummy"})
        target = target.rename({"mode": "feature1"})
        # -> target (feature1 x dummy)

        # Solve the symmetric eigenvalue problem
        eigensolver = Decomposer(
            n_modes=self._params["n_modes"], flip_signs=False, solver="full"
        )
        eigensolver.fit(target, dims=("feature1", "dummy"))
        U = eigensolver.U_
        # -> U (feature1 x mode)
        lbda = eigensolver.s_
        # -> lbda (mode)
        # U, lbda, ct = xr.apply_ufunc(
        #     np.linalg.svd,
        #     target,
        #     input_core_dims=[("feature1", "dummy")],
        #     output_core_dims=[("feature1", "mode"), ("mode",), ("mode", "dummy")],
        #     vectorize=False,
        #     dask="allowed",
        # )
        # Compute the filter patterns
        V = C0_sqrt_inv.rename({"mode": "mode1"}).dot(
            U.rename({"mode": "mode2"}), dims="feature1"
        )
        # -> V (mode1 x mode2)

        # Compute the optimally persistent patterns (OPPs)
        W = xr.dot(
            C0.rename({"feature2": "temp"}), V.rename({"mode1": "temp"}), dims="temp"
        )
        # -> W (feature1 x mode2)

        # Compute the time series of the optimally persistent patterns (OPPs)
        P = xr.dot(scores.rename({"mode": "mode1"}), V, dims="mode1")
        # -> P (sample x mode2)

        # Transform filter patterns and OPPs into original space
        V = xr.dot(comps.rename({"mode": "mode1"}), V, dims="mode1")
        # -> V (feature x mode2)

        W = xr.dot(comps.rename({"mode": "feature1"}), W, dims="feature1")
        # -> W (feature x mode2)

        # Rename dimensions
        U = U.rename({"feature1": feature_name})  # -> (feature x mode)
        V = V.rename({"mode2": "mode"})  # -> (feature x mode)
        W = W.rename({"mode2": "mode"})  # -> (feature x mode)
        P = P.rename({"mode2": "mode"})  # -> (sample x mode)
        scores = scores.rename({"mode": feature_name})  # -> (sample x feature)

        # Compute the norms of the scores
        norms = xr.apply_ufunc(
            np.linalg.norm,
            P,
            input_core_dims=[["sample"]],
            vectorize=False,
            dask="allowed",
            kwargs={"axis": -1},
        )

        # Store the results
        # NOTE: not sure if "scores" should be taken as input data here, "data" may be more correct -> to be verified
        self.data.add(name="input_data", data=scores, allow_compute=False)
        self.data.add(name="components", data=W, allow_compute=True)
        self.data.add(name="scores", data=P, allow_compute=True)
        self.data.add(name="norms", data=norms, allow_compute=True)
        self.data.add(name="filter_patterns", data=V, allow_compute=True)
        self.data.add(name="decorrelation_time", data=lbda, allow_compute=True)

        self.data.set_attrs(self.attrs)
        self._U = U  # store U for testing purposes of orthogonality
        self._C0 = C0  # store C0 for testing purposes of orthogonality
        return self

    def _transform_algorithm(self, X: DataArray) -> DataArray:
        raise NotImplementedError("OPA does not (yet) support transform()")

    def _inverse_transform_algorithm(self, scores) -> DataObject:
        raise NotImplementedError("OPA does not (yet) support inverse_transform()")

    def components(self) -> DataObject:
        """Return the optimally persistent patterns (OPPs)."""
        return super().components()

    def scores(self) -> DataArray:
        """Return the time series of the OPPs.

        The time series have a maximum decorrelation time that are uncorrelated with each other.
        """
        return super().scores()

    def decorrelation_time(self) -> DataArray:
        """Return the decorrelation time of the optimal persistence pattern (OPP)."""
        return self.data["decorrelation_time"]

    def filter_patterns(self) -> DataObject:
        """Return the filter patterns."""
        fps = self.data["filter_patterns"]
        return self.preprocessor.inverse_transform_components(fps)
