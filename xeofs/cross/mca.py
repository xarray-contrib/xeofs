import warnings
from typing import Sequence

import numpy as np

from ..utils.data_types import DataArray
from .cpcca import CPCCA, ComplexCPCCA, HilbertCPCCA


class MCA(CPCCA):
    """Maximum Covariance Analysis (MCA).

    MCA seeks to find paris of coupled patterns that maximize the squared
    covariance [1]_ [2]_ .

    This method solves the following optimization problem:

        :math:`\\max_{q_x, q_y} \\left( q_x^T X^T Y q_y \\right)`

    subject to the constraints:

        :math:`q_x^T q_x = 1, \\quad q_y^T q_y = 1`

    where :math:`X` and :math:`Y` are the input data matrices and :math:`q_x`
    and :math:`q_y` are the corresponding pattern vectors.

    Parameters
    ----------
    n_modes : int, default=2
        Number of modes to calculate.
    standardize : Squence[bool] | bool, default=False
        Whether to standardize the input data. Generally not recommended as
        standardization can be managed by the degree of whitening.
    use_coslat : Sequence[bool] | bool, default=False
        For data on a longitude-latitude grid, whether to correct for varying
        grid cell areas towards the poles by scaling each grid point with the
        square root of the cosine of its latitude.
    use_pca : Sequence[bool] | bool, default=False
        Whether to preprocess each field individually by reducing dimensionality
        through PCA. The cross-covariance matrix is then computed in the reduced
        principal component space.
    n_pca_modes : Sequence[int | float | str] | int | float | str, default=0.999
        Number of modes to retain during PCA preprocessing step. If int,
        specifies the exact number of modes; if float, specifies the fraction of
        variance to retain; if "all", all modes are retained.
    pca_init_rank_reduction : Sequence[float] | float, default=0.3
        Relevant when `use_pca=True` and `n_pca_modes` is a float. Specifies the
        initial fraction of rank reduction for faster PCA computation via
        randomized SVD.
    check_nans : Sequence[bool] | bool, default=True
        Whether to check for NaNs in the input data. Set to False for lazy model
        evaluation.
    compute : bool, default=True
        Whether to compute the model elements eagerly. If True, the following
        are computed sequentially: preprocessor scaler, optional NaN checks, SVD
        decomposition, scores, and components.
    random_state : numpy.random.Generator | int | None, default=None
        Seed for the random number generator.
    sample_name : str, default="sample"
        Name for the new sample dimension.
    feature_name : Sequence[str] | str, default="feature"
        Name for the new feature dimension.
    solver : {"auto", "full", "randomized"}
        Solver to use for the SVD computation.
    solver_kwargs : dict, default={}
        Additional keyword arguments passed to the SVD solver function.


    References
    ----------
    .. [1] Bretherton, C., Smith, C., Wallace, J., 1992. An intercomparison of
        methods for finding coupled patterns in climate data. Journal of climate
        5, 541–560.
    .. [2] Wilks, D. S. Statistical Methods in the Atmospheric Sciences.
        (Academic Press, 2019).
        doi:https://doi.org/10.1016/B978-0-12-815823-4.00011-0.

    Examples
    --------

    Perform MCA on two datasets on a regular longitude-latitude grid:

    >>> model = MCA(n_modes=5, use_coslat=True)
    >>> model.fit(X, Y, dim="time")

    """

    def __init__(
        self,
        n_modes: int = 2,
        standardize: Sequence[bool] | bool = False,
        use_coslat: Sequence[bool] | bool = False,
        check_nans: Sequence[bool] | bool = True,
        use_pca: Sequence[bool] | bool = True,
        n_pca_modes: Sequence[float | int | str] | float | int | str = 0.999,
        pca_init_rank_reduction: Sequence[float] | float = 0.3,
        compute: bool = True,
        sample_name: str = "sample",
        feature_name: Sequence[str] | str = "feature",
        solver: str = "auto",
        random_state: np.random.Generator | int | None = None,
        solver_kwargs: dict = {},
    ):
        CPCCA.__init__(
            self,
            n_modes=n_modes,
            alpha=[1.0, 1.0],
            standardize=standardize,
            use_coslat=use_coslat,
            check_nans=check_nans,
            use_pca=use_pca,
            n_pca_modes=n_pca_modes,
            pca_init_rank_reduction=pca_init_rank_reduction,
            compute=compute,
            sample_name=sample_name,
            feature_name=feature_name,
            solver=solver,
            random_state=random_state,
            solver_kwargs=solver_kwargs,
        )
        self.attrs.update({"model": "Maximum Covariance Analysis"})
        # Renove alpha from the inherited CPCCA serialization params because it is hard-coded for MCA
        self._params.pop("alpha")

    def covariance_fraction_CD95(self):
        """Get the covariance fraction (CF).

        Cheng and Dunkerton (1995) [3]_ define the CF as follows:

        .. math::
            CF_i = \\frac{\\sigma_i}{\\sum_{i=1}^{m} \\sigma_i}

        where `m` is the total number of modes and :math:`\\sigma_i` is the
        `ith` singular value of the covariance matrix.

        This implementation estimates the sum of singular values from the first
        `n` modes, therefore one should aim to retain as many modes as possible
        to get a good estimate of the covariance fraction.

        Note
        ----
        In MCA, the focus is on maximizing the *squared* covariance (SC). As a
        result, this quantity is preserved during decomposition - meaning the SC
        of both datasets remains unchanged before and after decomposition. Each
        mode explains a fraction of the total SC, and together, all modes can
        reconstruct the total SC of the cross-covariance matrix. However, the
        (non-squared) covariance is not invariant in MCA; it is not preserved by
        the individual modes and cannot be reconstructed from them.
        Consequently, the squared covariance fraction (SCF) is invariant in MCA
        and is typically used to assess the relative importance of each mode. In
        contrast, the convariance fraction (CF) is not invariant. Cheng and
        Dunkerton [3]_ introduced the CF to compare the relative importance of
        modes before and after Varimax rotation in MCA. Notably, when the data
        fields in MCA are identical, the CF corresponds to the explained
        variance ratio in Principal Component Analysis (PCA).

        References
        ----------
        .. [3] Cheng, X. & Dunkerton, T. J. Orthogonal Rotation of Spatial
            Patterns Derived from Singular Value Decomposition Analysis. J.
            Climate 8, 2631–2643 (1995).

        """
        # Check how sensitive the CF is to the number of modes
        cov_exp = self._covariance_explained_DC95()
        tot_var = self._total_covariance()
        cf = cov_exp[0] / cov_exp.cumsum()
        change_per_mode = cf.shift({"mode": 1}) - cf
        change_in_cf_in_last_mode = change_per_mode.isel(mode=-1)
        if change_in_cf_in_last_mode > 0.001:
            warnings.warn(
                "The curent estimate of CF is sensitive to the number of modes retained. Please increase `n_modes` for a better estimate."
            )
        cov_frac = cov_exp / tot_var
        cov_frac.name = "covariance_fraction"
        cov_frac.attrs.update(cov_exp.attrs)
        return cov_frac

    def _squared_covariance(self) -> DataArray:
        """Get the squared covariance.

        The squared covariance is given by the squared singular values of the
        covariance matrix:

        .. math::
            SC_i = \\sigma_i^2

        where :math:`\\sigma_i` is the `ith` singular value of the covariance
        matrix.

        """
        # only true for MCA, for alpha < 1 the sigmas become more and more correlation coefficients
        # either remove this one and provide it only for MCA child class, or use error formulation
        sc = self.data["squared_covariance"]
        sc.name = "squared_covariance"
        return sc

    def _covariance_explained_DC95(self) -> DataArray:
        """Get the covariance explained (CE) per mode according to CD95.

        References
        ----------
        Cheng, X. & Dunkerton, T. J. Orthogonal Rotation of Spatial Patterns
        Derived from Singular Value Decomposition Analysis. J. Climate 8,
        2631–2643 (1995).

        """
        cov_exp = self._squared_covariance() ** (0.5)
        cov_exp.name = "pseudo_explained_covariance"
        return cov_exp

    def _total_covariance(self) -> DataArray:
        """Get the total covariance.

        This measure follows the defintion of Cheng and Dunkerton (1995).
        Note that this measure is not an invariant in MCA.

        """
        pseudo_tot_cov = self._covariance_explained_DC95().sum()
        pseudo_tot_cov.name = "pseudo_total_covariance"
        return pseudo_tot_cov


class ComplexMCA(ComplexCPCCA, MCA):
    """Complex MCA.

    MCA applied to a complex-valued field obtained from a pair of variables such
    as the zonal and meridional components, :math:`U` and :math:`V`, of the wind
    field. Complex EOF analysis then maximizes the squared covariance between
    two datasets of the form

    .. math::
        Z_x = U_x + iV_x

    and

    .. math::
        Z_y = U_y + iV_y

    into a set of complex-valued components and PC scores.


    Parameters
    ----------
    n_modes : int, default=2
        Number of modes to calculate.
    standardize : Squence[bool] | bool, default=False
        Whether to standardize the input data. Generally not recommended as
        standardization can be managed by the degree of whitening.
    use_coslat : Sequence[bool] | bool, default=False
        For data on a longitude-latitude grid, whether to correct for varying
        grid cell areas towards the poles by scaling each grid point with the
        square root of the cosine of its latitude.
    use_pca : Sequence[bool] | bool, default=False
        Whether to preprocess each field individually by reducing dimensionality
        through PCA. The cross-covariance matrix is then computed in the reduced
        principal component space.
    n_pca_modes : Sequence[int | float | str] | int | float | str, default=0.999
        Number of modes to retain during PCA preprocessing step. If int,
        specifies the exact number of modes; if float, specifies the fraction of
        variance to retain; if "all", all modes are retained.
    pca_init_rank_reduction : Sequence[float] | float, default=0.3
        Relevant when `use_pca=True` and `n_pca_modes` is a float. Specifies the
        initial fraction of rank reduction for faster PCA computation via
        randomized SVD.
    check_nans : Sequence[bool] | bool, default=True
        Whether to check for NaNs in the input data. Set to False for lazy model
        evaluation.
    compute : bool, default=True
        Whether to compute the model elements eagerly. If True, the following
        are computed sequentially: preprocessor scaler, optional NaN checks, SVD
        decomposition, scores, and components.
    random_state : numpy.random.Generator | int | None, default=None
        Seed for the random number generator.
    sample_name : str, default="sample"
        Name for the new sample dimension.
    feature_name : Sequence[str] | str, default="feature"
        Name for the new feature dimension.
    solver : {"auto", "full", "randomized"}
        Solver to use for the SVD computation.
    solver_kwargs : dict, default={}
        Additional keyword arguments passed to the SVD solver function.

    Examples
    --------

    With two DataArrays `u_i` and `v_i` representing the zonal and meridional
    components of the wind field for two different regions :math:`x` and
    :math:`y`, construct

    >>> X = u_x + 1j * v_x
    >>> Y = u_y + 1j * v_y

    and fit the Complex MCA model:

    >>> model = ComplexMCA(n_modes=5)
    >>> model.fit(X, Y, "time")


    """

    def __init__(
        self,
        n_modes: int = 2,
        standardize: Sequence[bool] | bool = False,
        use_coslat: Sequence[bool] | bool = False,
        check_nans: Sequence[bool] | bool = True,
        use_pca: Sequence[bool] | bool = True,
        n_pca_modes: Sequence[float | int | str] | float | int | str = 0.999,
        pca_init_rank_reduction: Sequence[float] | float = 0.3,
        compute: bool = True,
        sample_name: str = "sample",
        feature_name: Sequence[str] | str = "feature",
        solver: str = "auto",
        random_state: np.random.Generator | int | None = None,
        solver_kwargs: dict = {},
    ):
        ComplexCPCCA.__init__(
            self,
            n_modes=n_modes,
            alpha=[1.0, 1.0],
            standardize=standardize,
            use_coslat=use_coslat,
            check_nans=check_nans,
            use_pca=use_pca,
            n_pca_modes=n_pca_modes,
            pca_init_rank_reduction=pca_init_rank_reduction,
            compute=compute,
            sample_name=sample_name,
            feature_name=feature_name,
            solver=solver,
            random_state=random_state,
            solver_kwargs=solver_kwargs,
        )
        self.attrs.update({"model": "Complex MCA"})
        # Renove alpha from the inherited CPCCA serialization params because it is hard-coded for MCA
        self._params.pop("alpha")


class HilbertMCA(HilbertCPCCA, ComplexMCA):
    """Hilbert MCA.

    Hilbert MCA [1]_ (aka Analytical SVD),  extends MCA by
    examining amplitude-phase relationships. It augments the input data with its
    Hilbert transform, creating a complex-valued field.

    This method solves the following optimization problem:

        :math:`\\max_{q_x, q_y} \\left( q_x^H X^H Y q_y \\right)`

    subject to the constraints:

        :math:`q_x^H q_x = 1, \\quad q_y^H q_y = 1`

    where :math:`H` denotes the conjugate transpose and :math:`X` and :math:`Y`
    are the augmented data matrices.

    An optional padding with exponentially decaying values can be applied prior
    to the Hilbert transform in order to mitigate the impact of spectral
    leakage.


    Parameters
    ----------
    n_modes : int, default=2
        Number of modes to calculate.
    padding : Sequence[str] | str | None, default="exp"
        Padding method for the Hilbert transform. Available options are: - None:
        no padding - "exp": exponential decay
    decay_factor : Sequence[float] | float, default=0.2
        Decay factor for the exponential padding.
    standardize : Squence[bool] | bool, default=False
        Whether to standardize the input data. Generally not recommended as
        standardization can be managed by the degree of whitening.
    use_coslat : Sequence[bool] | bool, default=False
        For data on a longitude-latitude grid, whether to correct for varying
        grid cell areas towards the poles by scaling each grid point with the
        square root of the cosine of its latitude.
    use_pca : Sequence[bool] | bool, default=False
        Whether to preprocess each field individually by reducing dimensionality
        through PCA. The cross-covariance matrix is computed in the reduced
        principal component space.
    n_pca_modes : Sequence[int | float | str] | int | float | str, default=0.999
        Number of modes to retain during PCA preprocessing step. If int,
        specifies the exact number of modes; if float, specifies the fraction of
        variance to retain; if "all", all modes are retained.
    pca_init_rank_reduction : Sequence[float] | float, default=0.3
        Relevant when `use_pca=True` and `n_pca_modes` is a float. Specifies the
        initial fraction of rank reduction for faster PCA computation via
        randomized SVD.
    check_nans : Sequence[bool] | bool, default=True
        Whether to check for NaNs in the input data. Set to False for lazy model
        evaluation.
    compute : bool, default=True
        Whether to compute the model elements eagerly. If True, the following
        are computed sequentially: preprocessor scaler, optional NaN checks, SVD
        decomposition, scores, and components.
    random_state : numpy.random.Generator | int | None, default=None
        Seed for the random number generator.
    sample_name : str, default="sample"
        Name for the new sample dimension.
    feature_name : Sequence[str] | str, default="feature"
        Name for the new feature dimension.
    solver : {"auto", "full", "randomized"}
        Solver to use for the SVD computation.
    solver_kwargs : dict, default={}
        Additional keyword arguments passed to the SVD solver function.

    References
    ----------
    .. [1] Elipot, S., Frajka-Williams, E., Hughes, C. W., Olhede, S. &
        Lankhorst, M. Observed Basin-Scale Response of the North Atlantic
        Meridional Overturning Circulation to Wind Stress Forcing. Journal of
        Climate 30, 2029–2054 (2017).



    Examples
    --------
    >>> model = HilbertMCA(n_modes=5)
    >>> model.fit(X, Y, "time")

    """

    def __init__(
        self,
        n_modes: int = 2,
        padding: Sequence[str] | str | None = "exp",
        decay_factor: Sequence[float] | float = 0.2,
        standardize: Sequence[bool] | bool = False,
        use_coslat: Sequence[bool] | bool = False,
        check_nans: Sequence[bool] | bool = True,
        use_pca: Sequence[bool] | bool = True,
        n_pca_modes: Sequence[float | int | str] | float | int | str = 0.999,
        pca_init_rank_reduction: Sequence[float] | float = 0.3,
        compute: bool = True,
        sample_name: str = "sample",
        feature_name: Sequence[str] | str = "feature",
        solver: str = "auto",
        random_state: np.random.Generator | int | None = None,
        solver_kwargs: dict = {},
    ):
        HilbertCPCCA.__init__(
            self,
            n_modes=n_modes,
            alpha=[1.0, 1.0],
            standardize=standardize,
            use_coslat=use_coslat,
            check_nans=check_nans,
            use_pca=use_pca,
            n_pca_modes=n_pca_modes,
            pca_init_rank_reduction=pca_init_rank_reduction,
            compute=compute,
            sample_name=sample_name,
            feature_name=feature_name,
            solver=solver,
            random_state=random_state,
            solver_kwargs=solver_kwargs,
            padding=padding,
            decay_factor=decay_factor,
        )
        self.attrs.update({"model": "Hilbert MCA"})
        # Renove alpha from the inherited CPCCA serialization params because it is hard-coded for MCA
        self._params.pop("alpha")
