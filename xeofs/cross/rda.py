from typing import Sequence

import numpy as np

from .cpcca import CPCCA, ComplexCPCCA, HilbertCPCCA


class RDA(CPCCA):
    """Redundancy Analysis (RDA).

    RDA seeks to find paris of coupled patterns that maximize the predictand
    variance [1]_ [2]_ .

    This method solves the following optimization problem:

        :math:`\\max_{q_x, q_y} \\left( q_x^T X^T Y q_y \\right)`

    subject to the constraints:

        :math:`q_x^T (X^TX) q_x = 1, \\quad q_y^T q_y = 1`

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
    .. [1] 1. Storch, H. von & Zwiers, F. W. Statistical Analysis in Climate
        Research. (Cambridge University Press (Virtual Publishing), 2003).
    .. [2] Wilks, D. S. Statistical Methods in the Atmospheric Sciences.
        (Academic Press, 2019).
        doi:https://doi.org/10.1016/B978-0-12-815823-4.00011-0.

    Examples
    --------

    Perform RDA on two datasets on a regular longitude-latitude grid:

    >>> model = RDA(n_modes=5, use_coslat=True)
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
            alpha=[0.0, 1.0],
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
        self.attrs.update({"model": "Redundancy Analysis"})
        # Renove alpha from the inherited CPCCA serialization params because it is hard-coded for RDA
        self._params.pop("alpha")


class ComplexRDA(ComplexCPCCA, RDA):
    """Complex RDA.

    RDA applied to a complex-valued field obtained from a pair of variables such
    as the zonal and meridional components, :math:`U` and :math:`V`, of the wind
    field. Complex RDA analysis then maximizes the correlation between
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

    and fit the Complex RDA model:

    >>> model = ComplexRDA(n_modes=5)
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
            alpha=[0.0, 1.0],
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
        self.attrs.update({"model": "Complex RDA"})
        # Renove alpha from the inherited CPCCA serialization params because it is hard-coded for RDA
        self._params.pop("alpha")


class HilbertRDA(HilbertCPCCA, ComplexRDA):
    """Hilbert RDA.

    Hilbert RDA  extends RDA by examining amplitude-phase relationships. It
    augments the input data with its Hilbert transform, creating a
    complex-valued field.

    This method solves the following optimization problem:

        :math:`\\max_{q_x, q_y} \\left( q_x^H X^H Y q_y \\right)`

    subject to the constraints:

        :math:`q_x^H (X^HX) q_x = 1, \\quad q_y^H q_y = 1`

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



    Examples
    --------
    >>> model = HilbertRDA(n_modes=5)
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
            alpha=[0.0, 1.0],
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
        self.attrs.update({"model": "Hilbert RDA"})
        # Renove alpha from the inherited CPCCA serialization params because it is hard-coded for RDA
        self._params.pop("alpha")
