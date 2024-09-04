import warnings

import numpy as np
import xarray as xr
from typing_extensions import Self

from ..linalg import total_variance
from ..preprocessing import Whitener
from ..utils.data_types import DataArray, DataObject
from ..utils.xarray_utils import argsort_dask
from .base_model_single_set import BaseModelSingleSet


class POP(BaseModelSingleSet):
    """Principal Oscillation Pattern (POP) analysis.

    POP analysis [1]_ [2]_ is a linear multivariate technique used to identify
    and describe dominant oscillatory modes in a dynamical system. POP analysis
    involves computing the eigenvalues and eigenvectors of the `feedback matrix`
    defined as

    .. math::
        A = C_1 C_0^{-1}

    where :math:`C_0` is the covariance matrix and :math:`C_1` is the lag-1
    covariance matrix of the input data. The eigenvectors of the feedback matrix
    are the POPs and the eigenvalues are related to the damping times and
    periods of the oscillatory modes.

    Parameters
    ----------
    n_modes: int, default=10
        Number of modes to calculate.
    center: bool, default=True
        Whether to center the input data.
    standardize: bool, default=False
        Whether to standardize the input data.
    use_coslat: bool, default=False
        Whether to use cosine of latitude for scaling.
    use_pca : bool, default=False
        If True, perform PCA to reduce the dimensionality of the data.
    n_pca_modes : int | float | str, default=0.999
        If int, specifies the number of modes to retain. If float, specifies the
        fraction of variance in the (whitened) data that should be explained by
        the retained modes. If "all", all modes are retained.
    init_rank_reduction : float, default=0.3
        Only relevant when `use_pca=True` and `n_modes` is a float, in which
        case it denotes the fraction of the initial rank to reduce the data to
        via PCA as a first guess before truncating the solution to the desired
        fraction of explained variance. This allows for faster computation of
        PCA via randomized SVD and avoids the need to compute the full SVD.
    sample_name: str, default="sample"
        Name of the sample dimension.
    feature_name: str, default="feature"
        Name of the feature dimension.
    check_nans : bool, default=True
        If True, remove full-dimensional NaN features from the data, check to
        ensure that NaN features match the original fit data during transform,
        and check for isolated NaNs. Note: this forces eager computation of dask
        arrays. If False, skip all NaN checks. In this case, NaNs should be
        explicitly removed or filled prior to fitting, or SVD will fail.
    compute : bool, default=True
        Whether to compute elements of the model eagerly, or to defer computation.
        If True, four pieces of the fit will be computed sequentially: 1) the
        preprocessor scaler, 2) optional NaN checks, 3) SVD decomposition, 4) scores
        and components.
    random_state : int, optional
        Seed for the random number generator.
    solver: {"auto", "full", "randomized"}, default="auto"
        Solver to use for the SVD computation.
    solver_kwargs: dict, default={}
        Additional keyword arguments to be passed to the SVD solver.

    References
    ----------
    .. [1] Hasselmann, K. PIPs and POPs: The reduction of complex dynamical systems using principal interaction and oscillation patterns. J. Geophys. Res. 93, 11015–11021 (1988).
    .. [2] von Storch, H., G. Bürger, R. Schnur, and J. von Storch, 1995:
    Principal Oscillation Patterns: A Review. J. Climate, 8, 377–400,
    https://doi.org/10.1175/1520-0442(1995)008<0377:POPAR>2.0.CO;2.


    Examples
    --------

    Perform POP analysis in PC space spanned by the first 10 modes:

    >>> pop = xe.single.POP(n_modes="all", use_pca=True, n_pca_modes=10)
    >>> pop.fit(X, "time)

    Get the POPs and associated time coefficients:

    >>> patterns = pop.components()
    >>> scores = pop.scores()

    Reconstruct the original data using a conjugate pair of POPs:

    >>> pop_pairs = scores.sel(mode=[1, 2])
    >>> X_rec = pop.inverse_transform(pop_pairs)

    """

    def __init__(
        self,
        n_modes: int = 2,
        center: bool = True,
        standardize: bool = False,
        use_coslat: bool = False,
        use_pca: bool = True,
        n_pca_modes: float | int = 0.999,
        pca_init_rank_reduction: float = 0.3,
        check_nans=True,
        sample_name: str = "sample",
        feature_name: str = "feature",
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
        self.attrs.update({"model": "Principal Oscillation Pattern analysis"})

        self.whitener = Whitener(
            alpha=1.0,
            use_pca=use_pca,
            n_modes=n_pca_modes,
            init_rank_reduction=pca_init_rank_reduction,
            sample_name=sample_name,
            feature_name=feature_name,
            compute_svd=compute,
            random_state=random_state,
            solver_kwargs=solver_kwargs,
        )

        self.sorted = False

    def get_serialization_attrs(self) -> dict:
        return dict(
            data=self.data,
            preprocessor=self.preprocessor,
            whitener=self.whitener,
            sorted=self.sorted,
        )

    def _np_solve_pop_system(self, X):
        # Feedack matrix
        A = X[1:].conj().T @ X[:-1] @ np.linalg.inv(X[:-1].conj().T @ X[:-1])

        # Compute POPs
        lbda, P = np.linalg.eig(A)

        # e-folding times /damping times
        tau = -1 / np.log(abs(lbda))

        # POP periods
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings(
                "ignore", "divide by zero encountered", RuntimeWarning
            )

            T = 2 * np.pi / np.angle(lbda)

        # POP (time) coefficients (Storch et al. 1995, equation 19)
        Z = self._np_compute_pop_coefficients(X, P)
        # Reconstruction of original data
        # Xrec = Z @ P.T
        # It seems that the signs of some columns of Xrec are flipped, probably

        return P, Z, lbda, T, tau

    def _np_compute_pop_coefficients(self, X, P):
        # POP (time) coefficients (Storch et al. 1995, equation 19)
        Z = np.empty((X.shape[0], P.shape[1]), dtype=complex)
        for i in range(P.shape[1]):
            p = P[:, i : i + 1]
            pr = p.real
            pi = p.imag

            M = np.array([[pr.T @ pr, pr.T @ pi], [pr.T @ pi, pi.T @ pi]]).squeeze()
            Minv = np.linalg.pinv(M)
            zri = Minv @ np.hstack([X @ pr, X @ pi]).T
            z = zri[0] + 1j * zri[1]
            Z[:, i] = z
        return Z

    def _fit_algorithm(self, X: DataArray) -> Self:
        sample_name = self.sample_name
        feature_name = self.feature_name

        # Transform in PC space
        X = self.whitener.fit_transform(X)

        P, Z, lbda, T, tau = xr.apply_ufunc(
            self._np_solve_pop_system,
            X,
            input_core_dims=[[sample_name, feature_name]],
            output_core_dims=[
                [feature_name, "mode"],
                [sample_name, "mode"],
                ["mode"],
                ["mode"],
                ["mode"],
            ],
            dask="allowed",
        )

        mode_coords = np.arange(1, P.mode.size + 1)
        P = P.assign_coords(mode=mode_coords)
        Z = Z.assign_coords(mode=mode_coords)
        lbda = lbda.assign_coords(mode=mode_coords)
        T = T.assign_coords(mode=mode_coords)
        tau = tau.assign_coords(mode=mode_coords)

        # Compute dynamical importance of each mode
        var_Z = Z.var(sample_name)
        norms = (var_Z) ** (0.5)

        # Compute total variance
        var_tot = total_variance(X, sample_name)

        # Reorder according to variance
        idx_modes_sorted = argsort_dask(norms, "mode")[::-1]  # type: ignore
        idx_modes_sorted.coords.update(norms.coords)

        P = self.whitener.inverse_transform_components(P)

        # Store the results
        self.data.add(X, "input_data", allow_compute=False)
        self.data.add(P, "components")
        self.data.add(Z, "scores")
        self.data.add(norms, "norms")
        self.data.add(lbda, "eigenvalues")
        self.data.add(tau, "damping_times")
        self.data.add(T, "periods")
        self.data.add(idx_modes_sorted, "idx_modes_sorted")
        self.data.add(var_tot, "total_variance")

        self.data.set_attrs(self.attrs)
        return self

    def _post_compute(self):
        """Leave sorting until after compute because it can't be done lazily."""
        self._sort_by_variance()

    def _sort_by_variance(self):
        """Re-sort the mode dimension of all data variables by variance explained."""
        if not self.sorted:
            for key in self.data.keys():
                if "mode" in self.data[key].dims and key != "idx_modes_sorted":
                    self.data[key] = (
                        self.data[key]
                        .isel(mode=self.data["idx_modes_sorted"].values)
                        .assign_coords(mode=self.data[key].mode)
                    )
        self.sorted = True

    def _transform_algorithm(self, X: DataArray) -> DataArray:
        sample_name = self.sample_name
        feature_name = self.feature_name

        P = self.data["components"]

        # Transform into PC spcae
        P = self.whitener.transform_components(P)
        X = self.whitener.transform(X)

        # Project the data
        Z = xr.apply_ufunc(
            self._np_compute_pop_coefficients,
            X,
            P,
            input_core_dims=[[sample_name, feature_name], [feature_name, "mode"]],
            output_core_dims=[[sample_name, "mode"]],
            dask="allowed",
        )
        Z.name = "scores"

        Z = self.whitener.inverse_transform_scores(Z)

        return Z

    def _inverse_transform_algorithm(self, scores: DataArray) -> DataArray:
        """Reconstruct the original data from transformed data.

        Parameters
        ----------
        scores: DataArray
            Transformed data to be reconstructed. This could be a subset
            of the `scores` data of a fitted model, or unseen data. Must
            have a 'mode' dimension.

        Returns
        -------
        data: DataObject
            Reconstructed data.

        """
        # Reconstruct the data
        P = self.data["components"].sel(mode=scores.mode)

        # Transform in PC space
        P = self.whitener.transform_components(P)

        reconstructed_data = xr.dot(scores, P, dims="mode")
        reconstructed_data.name = "reconstructed_data"

        # Inverse transform the data into physical space
        reconstructed_data = self.whitener.inverse_transform_data(reconstructed_data)

        return reconstructed_data

    def components(self) -> DataObject:
        """Return the POPs.

        The POPs are the eigenvectors of the feedback matrix.

        Returns
        -------
        components: DataObject
            Principal Oscillation Patterns (POPs).

        """
        return super().components(normalized=False)

    def scores(self, normalized: bool = False) -> DataArray:
        """Return the POP coefficients/scores.

        Parameters
        ----------
        normalized : bool, default=True
            Whether to normalize the scores by the L2 norm.

        Returns
        -------
        components: DataObject
            POP coefficients.

        """
        return super().scores(normalized=normalized)

    def eigenvalues(self) -> DataArray:
        """Return the eigenvalues of the feedback matrix.

        Returns
        -------
        DataArray
            Real or complex eigenvalues.

        """
        return self.data["eigenvalues"]

    def damping_times(self) -> DataArray:
        """Return the damping times of the feedback matrix.

        The damping times are defined as

        .. math::
            \\tau = -\\frac{1}{\\log(|\\lambda|)}

        where :math:`\\lambda` is the eigenvalue.

        Returns
        -------
        DataArray
            Damping times.

        """
        return self.data["damping_times"]

    def periods(self) -> DataArray:
        """Return the periods of the feedback matrix.

        For complex eigenvalues, the periods are defined as

        .. math::
            T = \\frac{2\\pi}{\\arg(\\lambda)}

        where :math:`\\lambda` is the eigenvalue. For real eigenvalues ``inf``
        is returned.

        Returns
        -------
        DataArray
            Periods.

        """
        return self.data["periods"]

    def components_amplitude(self) -> DataObject:
        """Return the amplitude of the POP components.

        The amplitude of the components are defined as

        .. math::
            A_{ij} = |C_{ij}|

        where :math:`C_{ij}` is the :math:`i`-th entry of the :math:`j`-th component and
        :math:`|\\cdot|` denotes the absolute value.


        Returns
        -------
        components_amplitude: DataObject
            Amplitude of the components of the fitted model.

        """
        amplitudes = abs(self.data["components"])

        amplitudes.name = "components_amplitude"
        return self.preprocessor.inverse_transform_components(amplitudes)

    def components_phase(self) -> DataObject:
        """Return the phase of the POP components.

        The phase of the components are defined as

        .. math::
            \\phi_{ij} = \\arg(C_{ij})

        where :math:`C_{ij}` is the :math:`i`-th entry of the :math:`j`-th component and
        :math:`\\arg(\\cdot)` denotes the argument of a complex number.

        Returns
        -------
        components_phase: DataObject
            Phase of the components of the fitted model.

        """
        comps = self.data["components"]
        comp_phase = xr.apply_ufunc(np.angle, comps, dask="allowed", keep_attrs=True)
        comp_phase.name = "components_phase"
        return self.preprocessor.inverse_transform_components(comp_phase)

    def scores_amplitude(self, normalized=True) -> DataArray:
        """Return the amplitude of the POP coefficients/scores.

        The amplitude of the scores are defined as

        .. math::
            A_{ij} = |S_{ij}|

        where :math:`S_{ij}` is the :math:`i`-th entry of the :math:`j`-th score and
        :math:`|\\cdot|` denotes the absolute value.

        Parameters
        ----------
        normalized : bool, default=True
            Whether to normalize the scores by the singular values.

        Returns
        -------
        scores_amplitude: DataObject
            Amplitude of the scores of the fitted model.

        """
        scores = self.data["scores"].copy()
        if normalized:
            scores = scores / self.data["norms"]

        amplitudes = abs(scores)
        amplitudes.name = "scores_amplitude"
        return self.preprocessor.inverse_transform_scores(amplitudes)

    def scores_phase(self) -> DataArray:
        """Return the phase of the POP coefficients/scores.

        The phase of the scores are defined as

        .. math::
            \\phi_{ij} = \\arg(S_{ij})

        where :math:`S_{ij}` is the :math:`i`-th entry of the :math:`j`-th score and
        :math:`\\arg(\\cdot)` denotes the argument of a complex number.

        Returns
        -------
        scores_phase: DataObject
            Phase of the scores of the fitted model.

        """
        scores = self.data["scores"]
        phases = xr.apply_ufunc(np.angle, scores, dask="allowed", keep_attrs=True)
        phases.name = "scores_phase"
        return self.preprocessor.inverse_transform_scores(phases)
