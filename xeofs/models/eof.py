from typing import Optional, Dict
from typing_extensions import Self
import numpy as np
import xarray as xr

from ._base_model import _BaseModel
from .decomposer import Decomposer
from ..utils.data_types import DataObject, DataArray, Dims
from ..utils.hilbert_transform import hilbert_transform
from ..utils.xarray_utils import total_variance as compute_total_variance


class EOF(_BaseModel):
    """EOF analysis.

    Empirical Orthogonal Functions (EOF) analysis, more commonly known
    as Principal Component Analysis (PCA).

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
    sample_name: str, default="sample"
        Name of the sample dimension.
    feature_name: str, default="feature"
        Name of the feature dimension.
    compute : bool, default=True
        Whether to compute elements of the model eagerly, or to defer computation.
        If True, four pieces of the fit will be computed sequentially: 1) the
        preprocessor scaler, 2) optional NaN checks, 3) SVD decomposition, 4) scores
        and components.
    verbose: bool, default=False
        Whether to show a progress bar when computing the decomposition.
    random_state : Optional[int], default=None
        Seed for the random number generator.
    solver: {"auto", "full", "randomized"}, default="auto"
        Solver to use for the SVD computation.
    solver_kwargs: dict, default={}
        Additional keyword arguments to be passed to the SVD solver.

    Examples
    --------
    >>> model = xe.models.EOF(n_modes=5)
    >>> model.fit(data)
    >>> scores = model.scores()

    """

    def __init__(
        self,
        n_modes: int = 2,
        center: bool = True,
        standardize: bool = False,
        use_coslat: bool = False,
        check_nans=True,
        sample_name: str = "sample",
        feature_name: str = "feature",
        compute: bool = True,
        verbose: bool = False,
        random_state: Optional[int] = None,
        solver: str = "auto",
        solver_kwargs: Dict = {},
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
            verbose=verbose,
            random_state=random_state,
            solver=solver,
            solver_kwargs=solver_kwargs,
            **kwargs,
        )
        self.attrs.update({"model": "EOF analysis"})

    def _fit_algorithm(self, data: DataArray) -> Self:
        sample_name = self.sample_name
        feature_name = self.feature_name

        # Compute the total variance
        total_variance = compute_total_variance(data, dim=sample_name)

        # Decompose the data
        decomposer = Decomposer(**self._decomposer_kwargs)
        decomposer.fit(data, dims=(sample_name, feature_name))

        singular_values = decomposer.s_
        components = decomposer.V_
        scores = decomposer.U_ * decomposer.s_
        scores.name = "scores"

        # Compute the explained variance per mode
        n_samples = data.coords[self.sample_name].size
        exp_var = singular_values**2 / (n_samples - 1)
        exp_var.name = "explained_variance"

        # Store the results
        self.data.add(data, "input_data", allow_compute=False)
        self.data.add(components, "components")
        self.data.add(scores, "scores")
        self.data.add(singular_values, "norms")
        self.data.add(exp_var, "explained_variance")
        self.data.add(total_variance, "total_variance")

        self.data.set_attrs(self.attrs)
        return self

    def _transform_algorithm(self, data: DataObject) -> DataArray:
        feature_name = self.preprocessor.feature_name

        components = self.data["components"]

        # Project the data
        projections = xr.dot(data, components, dims=feature_name)
        projections.name = "scores"

        return projections

    def _inverse_transform_algorithm(self, scores: DataObject) -> DataArray:
        """Reconstruct the original data from transformed data.

        Parameters
        ----------
        scores: DataObject
            Transformed data to be reconstructed. This could be a subset
            of the `scores` data of a fitted model, or unseen data. Must
            have a 'mode' dimension.

        Returns
        -------
        data: DataArray | Dataset | List[DataArray]
            Reconstructed data.

        """
        # Reconstruct the data
        comps = self.data["components"].sel(mode=scores.mode)

        reconstructed_data = xr.dot(comps.conj(), scores)
        reconstructed_data.name = "reconstructed_data"

        # Enforce real output
        reconstructed_data = reconstructed_data.real

        return reconstructed_data

    def components(self) -> DataObject:
        """Return the (EOF) components.

        The components in EOF anaylsis are the eigenvectors of the covariance/correlation matrix.
        Other names include the principal components or EOFs.

        Returns
        -------
        components: DataArray | Dataset | List[DataArray]
            Components of the fitted model.

        """
        return super().components()

    def scores(self, normalized: bool = True) -> DataArray:
        """Return the (PC) scores.

        The scores in EOF anaylsis are the projection of the data matrix onto the
        eigenvectors of the covariance matrix (or correlation) matrix.
        Other names include the principal component (PC) scores or just PCs.

        Parameters
        ----------
        normalized : bool, default=True
            Whether to normalize the scores by the L2 norm (singular values).

        Returns
        -------
        components: DataArray | Dataset | List[DataArray]
            Scores of the fitted model.

        """
        return super().scores(normalized=normalized)

    def singular_values(self) -> DataArray:
        """Return the singular values of the Singular Value Decomposition.

        Returns
        -------
        singular_values: DataArray
            Singular values obtained from the SVD.

        """
        return self.data["norms"]

    def explained_variance(self) -> DataArray:
        """Return explained variance.

        The explained variance :math:`\\lambda_i` is the variance explained
        by each mode. It is defined as

        .. math::
            \\lambda_i = \\frac{\\sigma_i^2}{N-1}

        where :math:`\\sigma_i` is the singular value of the :math:`i`-th mode and :math:`N` is the number of samples.
        Equivalently, :math:`\\lambda_i` is the :math:`i`-th eigenvalue of the covariance matrix.

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


class ComplexEOF(EOF):
    """Complex EOF analysis.

    The Complex EOF analysis [1]_ [2]_ [3]_ [4]_ (also known as Hilbert EOF analysis) applies a Hilbert transform
    to the data before performing the standard EOF analysis.
    The Hilbert transform is applied to each feature of the data individually.

    An optional padding with exponentially decaying values can be applied prior to
    the Hilbert transform in order to mitigate the impact of spectral leakage.

    Parameters
    ----------
    n_modes : int
        Number of modes to calculate.
    padding : str, optional
        Specifies the method used for padding the data prior to applying the Hilbert
        transform. This can help to mitigate the effect of spectral leakage.
        Currently, only 'exp' for exponential padding is supported. Default is 'exp'.
    decay_factor : float, optional
        Specifies the decay factor used in the exponential padding. This parameter
        is only used if padding='exp'. The recommended value typically ranges between 0.05 to 0.2
        but ultimately depends on the variability in the data.
        A smaller value (e.g. 0.05) is recommended for
        data with high variability, while a larger value (e.g. 0.2) is recommended
        for data with low variability. Default is 0.2.
    center: bool, default=True
        Whether to center the input data.
    standardize : bool
        Whether to standardize the input data.
    use_coslat : bool
        Whether to use cosine of latitude for scaling.
    sample_name: str, default="sample"
        Name of the sample dimension.
    feature_name: str, default="feature"
        Name of the feature dimension.
    compute : bool, default=True
        Whether to compute elements of the model eagerly, or to defer computation.
        If True, four pieces of the fit will be computed sequentially: 1) the
        preprocessor scaler, 2) optional NaN checks, 3) SVD decomposition, 4) scores
        and components.
    verbose: bool, default=False
        Whether to show a progress bar when computing the decomposition.
    random_state : Optional[int], default=None
        Seed for the random number generator.
    solver: {"auto", "full", "randomized"}, default="auto"
        Solver to use for the SVD computation.
    solver_kwargs: dict, default={}
        Additional keyword arguments to be passed to the SVD solver.
    solver_kwargs : dict, optional
        Additional keyword arguments to be passed to the SVD solver.

    References
    ----------
    .. [1] Rasmusson, E. M., Arkin, P. A., Chen, W.-Y. & Jalickee, J. B. Biennial variations in surface temperature over the United States as revealed by singular decomposition. Monthly Weather Review 109, 587–598 (1981).
    .. [2] Barnett, T. P. Interaction of the Monsoon and Pacific Trade Wind System at Interannual Time Scales Part I: The Equatorial Zone. Monthly Weather Review 111, 756–773 (1983).
    .. [3] Horel, J. Complex Principal Component Analysis: Theory and Examples. J. Climate Appl. Meteor. 23, 1660–1673 (1984).
    .. [4] Hannachi, A., Jolliffe, I. & Stephenson, D. Empirical orthogonal functions and related techniques in atmospheric science: A review. International Journal of Climatology 27, 1119–1152 (2007).

    Examples
    --------
    >>> model = ComplexEOF(n_modes=5, standardize=True)
    >>> model.fit(data)

    """

    def __init__(
        self,
        n_modes: int = 2,
        padding: str = "exp",
        decay_factor: float = 0.2,
        center: bool = True,
        standardize: bool = False,
        use_coslat: bool = False,
        check_nans: bool = True,
        sample_name: str = "sample",
        feature_name: str = "feature",
        compute: bool = True,
        verbose: bool = False,
        random_state: Optional[int] = None,
        solver: str = "auto",
        solver_kwargs: Dict = {},
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
            verbose=verbose,
            random_state=random_state,
            solver=solver,
            solver_kwargs=solver_kwargs,
            **kwargs,
        )
        self.attrs.update({"model": "Complex EOF analysis"})
        self._params.update({"padding": padding, "decay_factor": decay_factor})

    def _fit_algorithm(self, data: DataArray) -> Self:
        sample_name = self.sample_name
        feature_name = self.feature_name

        # Apply hilbert transform:
        padding = self._params["padding"]
        decay_factor = self._params["decay_factor"]
        data = hilbert_transform(
            data,
            dims=(sample_name, feature_name),
            padding=padding,
            decay_factor=decay_factor,
        )

        # Compute the total variance
        total_variance = compute_total_variance(data, dim=sample_name)

        # Decompose the complex data
        decomposer = Decomposer(**self._decomposer_kwargs)
        decomposer.fit(data)

        singular_values = decomposer.s_
        components = decomposer.V_
        scores = decomposer.U_ * decomposer.s_

        # Compute the explained variance per mode
        n_samples = data.coords[self.sample_name].size
        exp_var = singular_values**2 / (n_samples - 1)
        exp_var.name = "explained_variance"

        # Store the results
        self.data.add(data, "input_data", allow_compute=False)
        self.data.add(components, "components")
        self.data.add(scores, "scores")
        self.data.add(singular_values, "norms")
        self.data.add(exp_var, "explained_variance")
        self.data.add(total_variance, "total_variance")

        # Assign analysis-relevant meta data to the results
        self.data.set_attrs(self.attrs)
        return self

    def _transform_algorithm(self, data: DataArray) -> DataArray:
        raise NotImplementedError("Complex EOF does not support transform method.")

    def components_amplitude(self) -> DataObject:
        """Return the amplitude of the (EOF) components.

        The amplitude of the components are defined as

        .. math::
            A_{ij} = |C_{ij}|

        where :math:`C_{ij}` is the :math:`i`-th entry of the :math:`j`-th component and
        :math:`|\\cdot|` denotes the absolute value.

        Returns
        -------
        components_amplitude: DataArray | Dataset | List[DataArray]
            Amplitude of the components of the fitted model.

        """
        amplitudes = abs(self.data["components"])
        amplitudes.name = "components_amplitude"
        return self.preprocessor.inverse_transform_components(amplitudes)

    def components_phase(self) -> DataObject:
        """Return the phase of the (EOF) components.

        The phase of the components are defined as

        .. math::
            \\phi_{ij} = \\arg(C_{ij})

        where :math:`C_{ij}` is the :math:`i`-th entry of the :math:`j`-th component and
        :math:`\\arg(\\cdot)` denotes the argument of a complex number.

        Returns
        -------
        components_phase: DataArray | Dataset | List[DataArray]
            Phase of the components of the fitted model.

        """
        comps = self.data["components"]
        comp_phase = xr.apply_ufunc(np.angle, comps, dask="allowed", keep_attrs=True)
        comp_phase.name = "components_phase"
        return self.preprocessor.inverse_transform_components(comp_phase)

    def scores_amplitude(self, normalized=True) -> DataArray:
        """Return the amplitude of the (PC) scores.

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
        scores_amplitude: DataArray | Dataset | List[DataArray]
            Amplitude of the scores of the fitted model.

        """
        scores = self.data["scores"].copy()
        if normalized:
            scores = scores / self.data["norms"]

        amplitudes = abs(scores)
        amplitudes.name = "scores_amplitude"
        return self.preprocessor.inverse_transform_scores(amplitudes)

    def scores_phase(self) -> DataArray:
        """Return the phase of the (PC) scores.

        The phase of the scores are defined as

        .. math::
            \\phi_{ij} = \\arg(S_{ij})

        where :math:`S_{ij}` is the :math:`i`-th entry of the :math:`j`-th score and
        :math:`\\arg(\\cdot)` denotes the argument of a complex number.

        Returns
        -------
        scores_phase: DataArray | Dataset | List[DataArray]
            Phase of the scores of the fitted model.

        """
        scores = self.data["scores"]
        phases = xr.apply_ufunc(np.angle, scores, dask="allowed", keep_attrs=True)
        phases.name = "scores_phase"
        return self.preprocessor.inverse_transform_scores(phases)
