import numpy as np
import xarray as xr
from typing_extensions import Self

from ..data_container import DataContainer
from ..utils.data_types import DataArray
from .eof import EOF


class ExtendedEOF(EOF):
    """Extended EOF analysis.

    Extended EOF (EEOF) analysis [1]_ [2]_, often referred to as
    Multivariate/Multichannel Singular Spectrum Analysis, enhances
    traditional EOF analysis by identifying propagating signals or
    oscillations in multivariate datasets. This approach integrates the
    spatial correlation of EOFs with the temporal auto- and cross-correlation
    derived from the lagged covariance matrix.

    Parameters
    ----------
    n_modes : int
        Number of modes to be computed.
    tau : int
        Time delay used to construct a time-delayed version of the original time series.
    embedding : int
        Embedding dimension is the number of dimensions in the delay-coordinate space used to represent
        the dynamics of the system. It determines the number of delayed copies
        of the time series that are used to construct the delay-coordinate space.
    n_pca_modes : int, optional
        If provided, the input data is first preprocessed using PCA with the
        specified number of modes. The EEOF analysis is then performed on the
        resulting PCA scores. This approach can lead to important computational
        savings.
    **kwargs :
        Additional keyword arguments passed to the EOF model.

    References
    ----------
    .. [1] Weare, B. C. & Nasstrom, J. S. Examples of Extended Empirical Orthogonal Function Analyses. Monthly Weather Review 110, 481–485 (1982).
    .. [2] Broomhead, D. S. & King, G. P. Extracting qualitative dynamics from experimental data. Physica D: Nonlinear Phenomena 20, 217–236 (1986).


    Examples
    --------
    >>> from xeofs.single import EEOF
    >>> model = EEOF(n_modes=5, tau=1, embedding=20, n_pca_modes=20)
    >>> model.fit(data, dim=("time"))

    Retrieve the extended empirical orthogonal functions (EEOFs) and their explained variance:

    >>> eeofs = model.components()
    >>> exp_var = model.explained_variance()

    Retrieve the time-dependent coefficients corresponding to the EEOF modes:

    >>> scores = model.scores()
    """

    def __init__(
        self,
        n_modes: int,
        tau: int,
        embedding: int,
        n_pca_modes: int | None = None,
        center: bool = True,
        standardize: bool = False,
        use_coslat: bool = False,
        check_nans: bool = True,
        sample_name: str = "sample",
        feature_name: str = "feature",
        compute: bool = True,
        solver: str = "auto",
        random_state: int | None = None,
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
            solver=solver,
            random_state=random_state,
            solver_kwargs=solver_kwargs,
            **kwargs,
        )
        self.attrs.update({"model": "Extended EOF Analysis"})
        self._params.update(
            {"tau": tau, "embedding": embedding, "n_pca_modes": n_pca_modes}
        )

        # Initialize the DataContainer to store the results
        self.data = DataContainer()
        self.pca = (
            EOF(
                n_modes=n_pca_modes,
                center=True,
                standardize=False,
                use_coslat=False,
                compute=self._params["compute"],
                check_nans=False,
                sample_name=self.sample_name,
                feature_name=self.feature_name,
                solver_kwargs=self._params["solver_kwargs"],
            )
            if n_pca_modes
            else None
        )

    def _fit_algorithm(self, X: DataArray) -> Self:
        self.data.add(X.copy(), "input_data", allow_compute=False)

        # Preprocess the data using PCA
        if self.pca:
            self.pca.fit(X, dim=self.sample_name)
            X = self.pca.data["scores"]
            X = X.rename({"mode": self.feature_name})

        # Construct the time-delayed version of the original time series
        tau = self._params["tau"]
        embedding = self._params["embedding"]
        shift = np.arange(embedding) * tau
        X_extended = []
        for i in shift:
            X_extended.append(X.shift(sample=-i))
        X_extended = xr.concat(X_extended, dim="embedding")
        n_samples_cut = (embedding - 1) * tau
        X_extended = X_extended.isel(sample=slice(None, -n_samples_cut))
        X_extended.coords.update({"embedding": shift})

        # Perform standard PCA on extended data
        n_modes = self._params["n_modes"]
        model = EOF(
            n_modes=n_modes,
            center=True,
            standardize=False,
            use_coslat=False,
            compute=self._params["compute"],
            check_nans=False,
            sample_name=self.sample_name,
            feature_name=self.feature_name,
            solver=self._params["solver"],
            solver_kwargs=self._params["solver_kwargs"],
        )
        model.fit(X_extended, dim=self.sample_name)

        self.model = model
        self.data = model.data
        self.data["components"] = model.components()
        self.data["scores"] = model.scores(normalized=False)

        if self.pca:
            self.data["components"] = xr.dot(
                self.pca.data["components"].rename({"mode": "temp"}),
                self.data["components"].rename({"feature": "temp"}),
                dims="temp",
            )

        self.data.set_attrs(self.attrs)

        return self

    def _transform_algorithm(self, X):
        raise NotImplementedError("EEOF does currently not support transform")

    def _inverse_transform_algorithm(self, scores):
        # Reconstruct the data
        comps = self.data["components"].sel(mode=scores.mode, embedding=0, drop=True)

        reconstructed_data = xr.dot(comps.conj(), scores, optimize=True)
        reconstructed_data.name = "reconstructed_data"

        # Enforce real output
        reconstructed_data = reconstructed_data.real

        return reconstructed_data
