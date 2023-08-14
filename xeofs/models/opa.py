from typing import Optional

import xarray as xr
import numpy as np

from ._base_model import _BaseModel
from .eof import EOF
from .decomposer import Decomposer
from ..data_container.opa_data_container import OPADataContainer
from ..utils.data_types import AnyDataObject, DataArray


class OPA(_BaseModel):
    """Optimal Persistence Analysis (OPA).

    OPA identifies the optimal persistence patterns (OPP) with the
    largest decorrelation time in a time-varying field. Introduced by DelSole
    in 2001 [1]_, and further developed in 2006 [2]_, it's a method used to
    find patterns whose time series show strong persistence over time.

    Parameters
    ----------
    n_modes : int
        Number of optimal persistence patterns (OPP) to be computed.
    tau_max : int
        Maximum time lag for the computation of the covariance matrix.
    n_pca_modes : int
        Number of modes to be computed in the pre-processing step using EOF.

    References
    ----------
    .. [1] DelSole, T., 2001. Optimally Persistent Patterns in Time-Varying Fields. Journal of the Atmospheric Sciences 58, 1341–1356. https://doi.org/10.1175/1520-0469(2001)058<1341:OPPITV>2.0.CO;2
    .. [2] DelSole, T., 2006. Low-Frequency Variations of Surface Temperature in Observations and Simulations. Journal of Climate 19, 4487–4507. https://doi.org/10.1175/JCLI3879.1

    Examples
    --------
    >>> from xeofs.models import OPA
    >>> model = OPA(n_modes=10, tau_max=50, n_pca_modes=100)
    >>> model.fit(data, dim=("time"))

    Retrieve the optimal perstitence patterns (OPP) and their time series:

    >>> opp = model.components()
    >>> opp_ts = model.scores()

    Retrieve the decorrelation time of the optimal persistence patterns (OPP):

    >>> decorrelation_time = model.decorrelation_time()
    """

    def __init__(self, n_modes, tau_max, n_pca_modes, **kwargs):
        if n_modes > n_pca_modes:
            raise ValueError(
                f"n_modes must be smaller or equal to n_pca_modes (n_modes={n_modes}, n_pca_modes={n_pca_modes})"
            )
        super().__init__(n_modes=n_modes, **kwargs)
        self.attrs.update({"model": "OPA"})
        self._params.update({"tau_max": tau_max, "n_pca_modes": n_pca_modes})

        # Initialize the DataContainer to store the results
        self.data: OPADataContainer = OPADataContainer()

    def _Ctau(self, X, tau: int) -> DataArray:
        """Compute the time-lage covariance matrix C(tau) of the data X."""
        X0 = X.copy(deep=True)
        Xtau = X.shift(sample=-tau).dropna("sample")

        X0 = X0.rename({"mode": "feature1"})
        Xtau = Xtau.rename({"mode": "feature2"})
        return xr.dot(X0, Xtau, dims=["sample"]) / (Xtau.sample.size - 1)

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

    def fit(self, data: AnyDataObject, dim, weights: Optional[AnyDataObject] = None):
        # Preprocess the data
        input_data: DataArray = self.preprocessor.fit_transform(data, dim, weights)

        # Perform PCA as a pre-processing step
        pca = EOF(n_modes=self._params["n_pca_modes"], use_coslat=False)
        pca.fit(input_data, dim="sample")
        svals = pca.data.singular_values
        expvar = pca.data.explained_variance
        comps = pca.data.components * svals / np.sqrt(expvar)
        # -> comps (feature x mode)
        scores = pca.data.scores * np.sqrt(expvar)
        # -> scores (sample x mode)

        # Compute the covariance matrix with zero time lag
        C0 = self._Ctau(scores, 0)
        # -> C0 (feature1 x feature2)
        C0inv = self._compute_matrix_inverse(C0, dims=("feature1", "feature2"))
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
        decomposer = Decomposer(n_modes=C0.shape[0], flip_signs=False, solver="full")
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
        U = U.rename({"feature1": "feature"})  # -> (feature x mode)
        V = V.rename({"mode2": "mode"})  # -> (feature x mode)
        W = W.rename({"mode2": "mode"})  # -> (feature x mode)
        P = P.rename({"mode2": "mode"})  # -> (sample x mode)

        # Store the results
        self.data.set_data(
            input_data=scores.rename({"mode": "feature"}),
            components=W,
            scores=P,
            filter_patterns=V,
            decorrelation_time=lbda,
        )
        self.data.set_attrs(self.attrs)
        self._U = U  # store U for testing purposes of orthogonality
        self._C0 = C0  # store C0 for testing purposes of orthogonality

    def transform(self, data: AnyDataObject):
        raise NotImplementedError()

    def inverse_transform(self, mode):
        raise NotImplementedError()

    def components(self) -> AnyDataObject:
        """Return the optimal persistence pattern (OPP)."""
        return super().components()

    def scores(self) -> DataArray:
        """Return the time series of the optimal persistence pattern (OPP).

        The time series have a maximum decorrelation time that are uncorrelated with each other.
        """
        return super().scores()

    def decorrelation_time(self) -> DataArray:
        """Return the decorrelation time of the optimal persistence pattern (OPP)."""
        return self.data.decorrelation_time

    def filter_patterns(self) -> DataArray:
        """Return the filter patterns."""
        fps = self.data.filter_patterns
        return self.preprocessor.inverse_transform_components(fps)
