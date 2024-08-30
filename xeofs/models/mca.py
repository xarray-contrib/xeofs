import warnings

from ..utils.data_types import DataArray
from .cpcca import ComplexCPCCA, ContinuumPowerCCA


class MCA(ContinuumPowerCCA):
    """Maximum Covariance Analysis (MCA).

    MCA [1]_ [2]_ seeks to find paris of coupled patterns that maimize the
    squared covariance.

    This method solves the following optimization problem:

        :math:`\\max_{q_x, q_y} \\left( q_x^T X^T Y q_y \\right)`

    subject to the constraints:

        :math:`q_x^T q_x = 1, \\quad q_y^T q_y = 1`

    where :math:`X` and :math:`Y` are the input data matrices and :math:`q_x`
    and :math:`q_y` are the corresponding pattern vectors.

    Parameters
    ----------
    n_modes: int, default=2
        Number of modes to calculate.
    center: bool, default=True
        Whether to center the input data.
    standardize: bool, default=False
        Whether to standardize the input data.
    use_coslat: bool, default=False
        Whether to use cosine of latitude for scaling.
    n_pca_modes: int, default=None
        The number of principal components to retain during the PCA preprocessing
        step applied to both data sets prior to executing MCA.
        If set to None, PCA preprocessing will be bypassed, and the MCA will be performed on the original datasets.
        Specifying an integer value greater than 0 for `n_pca_modes` will trigger the PCA preprocessing, retaining
        only the specified number of principal components. This reduction in dimensionality can be especially beneficial
        when dealing with high-dimensional data, where computing the cross-covariance matrix can become computationally
        intensive or in scenarios where multicollinearity is a concern.
    compute : bool, default=True
        Whether to compute elements of the model eagerly, or to defer computation.
        If True, four pieces of the fit will be computed sequentially: 1) the
        preprocessor scaler, 2) optional NaN checks, 3) SVD decomposition, 4) scores
        and components.
    sample_name: str, default="sample"
        Name of the new sample dimension.
    feature_name: str, default="feature"
        Name of the new feature dimension.
    solver: {"auto", "full", "randomized"}, default="auto"
        Solver to use for the SVD computation.
    random_state: int, default=None
        Seed for the random number generator.
    solver_kwargs: dict, default={}
        Additional keyword arguments passed to the SVD solver function.


    References
    ----------
    .. [1] Bretherton, C., Smith, C., Wallace, J., 1992. An intercomparison of methods for finding coupled patterns in climate data. Journal of climate 5, 541–560.
    .. [2] Wilks, D. S. Statistical Methods in the Atmospheric Sciences. (Academic Press, 2019). doi:https://doi.org/10.1016/B978-0-12-815823-4.00011-0.

    Examples
    --------
    >>> model = MCA(n_modes=5, standardize=True)
    >>> model.fit(data1, data2)
    """

    def __init__(
        self,
        n_modes: int = 2,
        center: bool = True,
        standardize: bool = False,
        use_coslat: bool = False,
        check_nans: bool = True,
        use_pca: bool = True,
        n_pca_modes: int | float = 0.999,
        compute: bool = True,
        sample_name: str = "sample",
        feature_name: str = "feature",
        solver: str = "auto",
        random_state: int | None = None,
        solver_kwargs: dict = {},
        **kwargs,
    ):
        # NOTE: **kwargs is only used here to catch any additional arguments that may be passed during deserialization. During .serialize(), the MCA model stores all model parameters from the parent class ContinuumPowerCCA. During deserialization, `use_pca` and `alpha` are passed to the child class MCA where there are not present and would raise an error. To avoid this, we catch all additional arguments here and ignore them.
        super().__init__(
            n_modes=n_modes,
            alpha=[1.0, 1.0],
            standardize=standardize,
            use_coslat=use_coslat,
            check_nans=check_nans,
            use_pca=use_pca,
            n_pca_modes=n_pca_modes,
            compute=compute,
            sample_name=sample_name,
            feature_name=feature_name,
            solver=solver,
            random_state=random_state,
            solver_kwargs=solver_kwargs,
        )
        self.attrs.update({"model": "Maximum Covariance Analysis"})

    def _squared_covariance(self) -> DataArray:
        """Get the squared covariance.

        The squared covariance is given by the
        squared singular values of the covariance matrix:

        .. math::
            SC_i = \\sigma_i^2

        where :math:`\\sigma_i` is the `ith` singular value of the covariance matrix.

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
        .. Cheng, X. & Dunkerton, T. J. Orthogonal Rotation of Spatial Patterns Derived from Singular Value Decomposition Analysis. J. Climate 8, 2631–2643 (1995).

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

    def covariance_fraction_CD95(self):
        """Get the covariance fraction (CF).

        Cheng and Dunkerton (1995) [3]_ define the CF as follows:

        .. math::
            CF_i = \\frac{\\sigma_i}{\\sum_{i=1}^{m} \\sigma_i}

        where `m` is the total number of modes and :math:`\\sigma_i` is the
        `ith` singular value of the covariance matrix.

        In this implementation the sum of singular values is estimated from the
        first `n` modes, therefore one should aim to retain as many modes as
        possible to get a good estimate of the covariance fraction.

        Note
        ----
        Since CPCCA maximizes *squared* covariance (SC), it is this quantity that is going to be preserved during the decomposition, i.e. the SC of both data sets is the same before and after decomposition. Put differently, each mode explains a fraction of the total SC, and all modes together allow you to reconstruct the total SC of the cross-covariance matrix. The (non-squared) covariance is not an invariant under CPCCA. That is, the quantity is not preserved by the individual modes we obtain and cannnot reconstructed from them.    It is crucial to distinguish the covariance fraction (CF) from the
        squared covariance fraction (SCF). The SCF is invariant in CPCCA and is
        typically used to evaluate the relative importance of each mode. In
        contrast, the CF is not invariant. Cheng and Dunkerton introduced the CF
        to compare the relative importance of modes before and after Varimax
        rotation in MCA. Notably, when both data fields in MCA are identical,
        the CF is equivalent to the explained variance ratio in Principal
        Component Analysis (PCA).

        References
        ----------
        .. [3] Cheng, X. & Dunkerton, T. J. Orthogonal Rotation of Spatial Patterns Derived from Singular Value Decomposition Analysis. J. Climate 8, 2631–2643 (1995).
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


class ComplexMCA(ComplexCPCCA, MCA):
    """Complex MCA.

    Complex MCA [1]_ (aka Analytical SVD or Hilbert MCA),  extends classical CPCCA by examining
    amplitude-phase relationships. It augments the input data with its Hilbert
    transform, creating a complex-valued field.

    This method solves the following optimization problem:

        :math:`\\max_{q_x, q_y} \\left( q_x^H X^H Y q_y \\right)`

    subject to the constraints:

        :math:`q_x^H q_x = 1, \\quad q_y^H q_y = 1`

    where :math:`H` denotes the conjugate transpose and :math:`X` and :math:`Y` are
    the augmented data matrices.

    An optional padding with exponentially decaying values can be applied prior
    to the Hilbert transform in order to mitigate the impact of spectral
    leakage.


    Parameters
    ----------
    n_modes : int, default=2
        Number of modes to calculate.
    padding : Sequence[str] | str | None, default="exp"
        Padding method for the Hilbert transform. Available options are:
        - None: no padding
        - "exp": exponential decay
    decay_factor : Sequence[float] | float, default=0.2
        Decay factor for the exponential padding.
    standardize : Squence[bool] | bool, default=False
        Whether to standardize the input data. Generally not recommended as standardization can be managed by the degree of whitening.
    use_coslat : Sequence[bool] | bool, default=False
        For data on a longitude-latitude grid, whether to correct for varying grid cell areas towards the poles by scaling each grid point with the square root of the cosine of its latitude.
    use_pca : Sequence[bool] | bool, default=False
        Whether to preprocess each field individually by reducing dimensionality through PCA. The cross-covariance matrix is computed in the reduced principal component space.
    n_pca_modes : Sequence[int | float | str] | int | float | str, default=0.999
        Number of modes to retain during PCA preprocessing step. If int, specifies the exact number of modes; if float, specifies the fraction of variance to retain; if "all", all modes are retained.
    pca_init_rank_reduction : Sequence[float] | float, default=0.3
        Relevant when `use_pca=True` and `n_pca_modes` is a float. Specifies the initial fraction of rank reduction for faster PCA computation via randomized SVD.
    check_nans : Sequence[bool] | bool, default=True
        Whether to check for NaNs in the input data. Set to False for lazy model evaluation.
    compute : bool, default=True
        Whether to compute the model elements eagerly. If True, the following are computed sequentially: preprocessor scaler, optional NaN checks, SVD decomposition, scores, and components.
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
    .. [1] Elipot, S., Frajka-Williams, E., Hughes, C. W., Olhede, S. & Lankhorst, M. Observed Basin-Scale Response of the North Atlantic Meridional Overturning Circulation to Wind Stress Forcing. Journal of Climate 30, 2029–2054 (2017).


    """

    def __init__(
        self,
        n_modes: int = 2,
        padding: str = "exp",
        decay_factor: float = 0.2,
        standardize: bool = False,
        use_coslat: bool = False,
        check_nans: bool = True,
        use_pca: bool = True,
        n_pca_modes: int | float = 0.999,
        compute: bool = True,
        sample_name: str = "sample",
        feature_name: str = "feature",
        solver: str = "auto",
        random_state: int | None = None,
        solver_kwargs: dict = {},
        **kwargs,
    ):
        super().__init__(
            n_modes=n_modes,
            alpha=[1.0, 1.0],
            standardize=standardize,
            use_coslat=use_coslat,
            check_nans=check_nans,
            use_pca=use_pca,
            n_pca_modes=n_pca_modes,
            compute=compute,
            sample_name=sample_name,
            feature_name=feature_name,
            solver=solver,
            random_state=random_state,
            solver_kwargs=solver_kwargs,
            padding=padding,
            decay_factor=decay_factor,
        )
        self.attrs.update({"model": "Complex MCA"})


# class MCA(_BaseCrossModel):
#     """Maximum Covariance Analyis.

#     MCA is a statistical method that finds patterns of maximum covariance between two datasets.

#     Parameters
#     ----------
#     n_modes: int, default=2
#         Number of modes to calculate.
#     center: bool, default=True
#         Whether to center the input data.
#     standardize: bool, default=False
#         Whether to standardize the input data.
#     use_coslat: bool, default=False
#         Whether to use cosine of latitude for scaling.
#     n_pca_modes: int, default=None
#         The number of principal components to retain during the PCA preprocessing
#         step applied to both data sets prior to executing MCA.
#         If set to None, PCA preprocessing will be bypassed, and the MCA will be performed on the original datasets.
#         Specifying an integer value greater than 0 for `n_pca_modes` will trigger the PCA preprocessing, retaining
#         only the specified number of principal components. This reduction in dimensionality can be especially beneficial
#         when dealing with high-dimensional data, where computing the cross-covariance matrix can become computationally
#         intensive or in scenarios where multicollinearity is a concern.
#     compute : bool, default=True
#         Whether to compute elements of the model eagerly, or to defer computation.
#         If True, four pieces of the fit will be computed sequentially: 1) the
#         preprocessor scaler, 2) optional NaN checks, 3) SVD decomposition, 4) scores
#         and components.
#     sample_name: str, default="sample"
#         Name of the new sample dimension.
#     feature_name: str, default="feature"
#         Name of the new feature dimension.
#     solver: {"auto", "full", "randomized"}, default="auto"
#         Solver to use for the SVD computation.
#     random_state: int, default=None
#         Seed for the random number generator.
#     solver_kwargs: dict, default={}
#         Additional keyword arguments passed to the SVD solver function.

#     Notes
#     -----
#     MCA is similar to Principal Component Analysis (PCA) and Canonical Correlation Analysis (CCA),
#     but while PCA finds modes of maximum variance and CCA finds modes of maximum correlation,
#     MCA finds modes of maximum covariance. See [1]_ [2]_ for more details.

#     References
#     ----------
#     .. [1] Bretherton, C., Smith, C., Wallace, J., 1992. An intercomparison of methods for finding coupled patterns in climate data. Journal of climate 5, 541–560.
#     .. [2] Cherry, S., 1996. Singular value decomposition analysis and canonical correlation analysis. Journal of Climate 9, 2003–2009.

#     Examples
#     --------
#     >>> model = MCA(n_modes=5, standardize=True)
#     >>> model.fit(data1, data2)

#     """

#     def __init__(
#         self,
#         n_modes: int = 2,
#         center: bool = True,
#         standardize: bool = False,
#         use_coslat: bool = False,
#         check_nans: bool = True,
#         n_pca_modes: Optional[int] = None,
#         compute: bool = True,
#         sample_name: str = "sample",
#         feature_name: str = "feature",
#         solver: str = "auto",
#         random_state: Optional[int] = None,
#         solver_kwargs: dict = {},
#         **kwargs,
#     ):
#         super().__init__(
#             n_modes=n_modes,
#             center=center,
#             standardize=standardize,
#             use_coslat=use_coslat,
#             check_nans=check_nans,
#             n_pca_modes=n_pca_modes,
#             compute=compute,
#             sample_name=sample_name,
#             feature_name=feature_name,
#             solver=solver,
#             random_state=random_state,
#             solver_kwargs=solver_kwargs,
#             **kwargs,
#         )
#         self.attrs.update({"model": "MCA"})

#     def _compute_cross_covariance_matrix(self, X1, X2):
#         """Compute the cross-covariance matrix of two data objects.

#         Note: It is assumed that the data objects are centered.

#         """
#         sample_name = self.sample_name
#         n_samples = X1.coords[sample_name].size
#         if X1.coords[sample_name].size != X2.coords[sample_name].size:
#             err_msg = "The two data objects must have the same number of samples."
#             raise ValueError(err_msg)

#         return xr.dot(X1.conj(), X2, dims=sample_name) / (n_samples - 1)

#     def _fit_algorithm(
#         self,
#         data1: DataArray,
#         data2: DataArray,
#     ) -> Self:
#         sample_name = self.sample_name
#         feature_name = self.feature_name

#         # Initialize the SVD decomposer
#         decomposer = Decomposer(**self._decomposer_kwargs)

#         # Perform SVD on PCA-reduced data
#         if (self.pca1 is not None) and (self.pca2 is not None):
#             # Fit the PCA models
#             self.pca1.fit(data1, dim=sample_name)
#             self.pca2.fit(data2, dim=sample_name)
#             # Get the PCA scores
#             pca_scores1 = self.pca1.data["scores"] * self.pca1.singular_values()
#             pca_scores2 = self.pca2.data["scores"] * self.pca2.singular_values()
#             # Compute the cross-covariance matrix of the PCA scores
#             pca_scores1 = pca_scores1.rename({"mode": "feature1"})
#             pca_scores2 = pca_scores2.rename({"mode": "feature2"})
#             cov_matrix = self._compute_cross_covariance_matrix(pca_scores1, pca_scores2)

#             # Perform the SVD
#             decomposer.fit(cov_matrix, dims=("feature1", "feature2"))
#             V1 = decomposer.U_  # left singular vectors (feature1 x mode)
#             V2 = decomposer.V_  # right singular vectors (feature2 x mode)

#             # left and right PCA eigenvectors (feature x mode)
#             V1pre = self.pca1.data["components"]
#             V2pre = self.pca2.data["components"]

#             # Compute the singular vectors
#             V1pre = V1pre.rename({"mode": "feature1"})
#             V2pre = V2pre.rename({"mode": "feature2"})
#             singular_vectors1 = xr.dot(V1pre, V1, dims="feature1")
#             singular_vectors2 = xr.dot(V2pre, V2, dims="feature2")

#         # Perform SVD directly on data
#         else:
#             # Rename feature and associated dimensions of data objects to avoid index conflicts
#             dim_renamer1 = DimensionRenamer(feature_name, "1")
#             dim_renamer2 = DimensionRenamer(feature_name, "2")
#             data1_temp = dim_renamer1.fit_transform(data1)
#             data2_temp = dim_renamer2.fit_transform(data2)
#             # Compute the cross-covariance matrix
#             cov_matrix = self._compute_cross_covariance_matrix(data1_temp, data2_temp)

#             # Perform the SVD
#             decomposer.fit(cov_matrix, dims=("feature1", "feature2"))
#             singular_vectors1 = decomposer.U_
#             singular_vectors2 = decomposer.V_

#             # Rename the singular vectors
#             singular_vectors1 = dim_renamer1.inverse_transform(singular_vectors1)
#             singular_vectors2 = dim_renamer2.inverse_transform(singular_vectors2)

#         # Store the results
#         singular_values = decomposer.s_

#         # Compute total squared variance
#         squared_covariance = singular_values**2
#         total_squared_covariance = (abs(cov_matrix) ** 2).sum()

#         norm1 = np.sqrt(singular_values)
#         norm2 = np.sqrt(singular_values)

#         # Index of the sorted squared covariance
#         idx_sorted_modes = argsort_dask(squared_covariance, "mode")[::-1]
#         idx_sorted_modes.coords.update(squared_covariance.coords)

#         # Project the data onto the singular vectors
#         scores1 = xr.dot(data1, singular_vectors1, dims=feature_name) / norm1
#         scores2 = xr.dot(data2, singular_vectors2, dims=feature_name) / norm2

#         self.data.add(name="input_data1", data=data1, allow_compute=False)
#         self.data.add(name="input_data2", data=data2, allow_compute=False)
#         self.data.add(name="components1", data=singular_vectors1)
#         self.data.add(name="components2", data=singular_vectors2)
#         self.data.add(name="scores1", data=scores1)
#         self.data.add(name="scores2", data=scores2)
#         self.data.add(name="squared_covariance", data=squared_covariance)
#         self.data.add(name="total_squared_covariance", data=total_squared_covariance)
#         self.data.add(name="idx_modes_sorted", data=idx_sorted_modes)
#         self.data.add(name="norm1", data=norm1)
#         self.data.add(name="norm2", data=norm2)

#         # Assign analysis-relevant meta data
#         self.data.set_attrs(self.attrs)
#         return self

#     def transform(
#         self, data1: Optional[DataObject] = None, data2: Optional[DataObject] = None
#     ) -> Sequence[DataArray]:
#         """Get the expansion coefficients of "unseen" data.

#         The expansion coefficients are obtained by projecting data onto the singular vectors.

#         Parameters
#         ----------
#         data1: DataArray | Dataset | List[DataArray]
#             Left input data. Must be provided if `data2` is not provided.
#         data2: DataArray | Dataset | List[DataArray]
#             Right input data. Must be provided if `data1` is not provided.

#         Returns
#         -------
#         scores1: DataArray | Dataset | List[DataArray]
#             Left scores.
#         scores2: DataArray | Dataset | List[DataArray]
#             Right scores.

#         """
#         return super().transform(data1, data2)

#     def _transform_algorithm(
#         self, data1: Optional[DataArray] = None, data2: Optional[DataArray] = None
#     ) -> dict[str, DataArray]:
#         results = {}
#         if data1 is not None:
#             # Project data onto singular vectors
#             comps1 = self.data["components1"]
#             norm1 = self.data["norm1"]
#             scores1 = xr.dot(data1, comps1) / norm1
#             results["data1"] = scores1

#         if data2 is not None:
#             # Project data onto singular vectors
#             comps2 = self.data["components2"]
#             norm2 = self.data["norm2"]
#             scores2 = xr.dot(data2, comps2) / norm2
#             results["data2"] = scores2

#         return results

#     def _inverse_transform_algorithm(
#         self, scores1: DataArray, scores2: DataArray
#     ) -> Tuple[DataArray, DataArray]:
#         """Reconstruct the original data from transformed data.

#         Parameters
#         ----------
#         scores1: DataArray
#             Transformed left field data to be reconstructed. This could be
#             a subset of the `scores` data of a fitted model, or unseen data.
#             Must have a 'mode' dimension.
#         scores2: DataArray
#             Transformed right field data to be reconstructed. This could be
#             a subset of the `scores` data of a fitted model, or unseen data.
#             Must have a 'mode' dimension.

#         Returns
#         -------
#         Xrec1: DataArray
#             Reconstructed data of left field.
#         Xrec2: DataArray
#             Reconstructed data of right field.

#         """
#         # Singular vectors
#         comps1 = self.data["components1"].sel(mode=scores1.mode)
#         comps2 = self.data["components2"].sel(mode=scores2.mode)

#         # Norms
#         norm1 = self.data["norm1"].sel(mode=scores1.mode)
#         norm2 = self.data["norm2"].sel(mode=scores2.mode)

#         # Reconstruct the data
#         data1 = xr.dot(scores1, comps1.conj() * norm1, dims="mode")
#         data2 = xr.dot(scores2, comps2.conj() * norm2, dims="mode")

#         return data1, data2

#     def squared_covariance(self):
#         """Get the squared covariance.

#         The squared covariance corresponds to the explained variance in PCA and is given by the
#         squared singular values of the covariance matrix.

#         """
#         return self.data["squared_covariance"]

#     def squared_covariance_fraction(self):
#         """Calculate the squared covariance fraction (SCF).

#         The SCF is a measure of the proportion of the total squared covariance that is explained by each mode `i`. It is computed
#         as follows:

#         .. math::
#             SCF_i = \\frac{\\sigma_i^2}{\\sum_{i=1}^{m} \\sigma_i^2}

#         where `m` is the total number of modes and :math:`\\sigma_i` is the `ith` singular value of the covariance matrix.

#         """
#         return self.data["squared_covariance"] / self.data["total_squared_covariance"]

#     def singular_values(self):
#         """Get the singular values of the cross-covariance matrix."""
#         singular_values = xr.apply_ufunc(
#             np.sqrt,
#             self.data["squared_covariance"],
#             dask="allowed",
#             vectorize=False,
#             keep_attrs=True,
#         )
#         singular_values.name = "singular_values"
#         return singular_values

#     def total_covariance(self) -> DataArray:
#         """Get the total covariance.

#         This measure follows the defintion of Cheng and Dunkerton (1995).
#         Note that this measure is not an invariant in MCA.

#         """
#         tot_cov = self.singular_values().sum()
#         tot_cov.attrs.update(self.singular_values().attrs)
#         tot_cov.name = "total_covariance"
#         return tot_cov

#     def covariance_fraction(self):
#         """Get the covariance fraction (CF).

#         Cheng and Dunkerton (1995) define the CF as follows:

#         .. math::
#             CF_i = \\frac{\\sigma_i}{\\sum_{i=1}^{m} \\sigma_i}

#         where `m` is the total number of modes and :math:`\\sigma_i` is the
#         `ith` singular value of the covariance matrix.

#         In this implementation the sum of singular values is estimated from
#         the first `n` modes, therefore one should aim to retain as many
#         modes as possible to get a good estimate of the covariance fraction.

#         Note
#         ----
#         It is important to differentiate the CF from the squared covariance fraction (SCF). While the SCF is an invariant quantity in MCA, the CF is not.
#         Therefore, the SCF is used to assess the relative importance of each mode. Cheng and Dunkerton (1995) introduced the CF in the context of
#         Varimax-rotated MCA to compare the relative importance of each mode before and after rotation. In the special case of both data fields in MCA being identical,
#         the CF is equivalent to the explained variance ratio in EOF analysis.

#         """
#         # Check how sensitive the CF is to the number of modes
#         svals = self.singular_values()
#         tot_var = self.total_covariance()
#         cf = svals[0] / svals.cumsum()
#         change_per_mode = cf.shift({"mode": 1}) - cf
#         change_in_cf_in_last_mode = change_per_mode.isel(mode=-1)
#         if change_in_cf_in_last_mode > 0.001:
#             print(
#                 "Warning: CF is sensitive to the number of modes retained. Please increase `n_modes` for a better estimate."
#             )
#         cov_frac = svals / tot_var
#         cov_frac.name = "covariance_fraction"
#         cov_frac.attrs.update(svals.attrs)
#         return cov_frac

#     def components(self):
#         """Return the singular vectors of the left and right field.

#         Returns
#         -------
#         components1: DataArray | Dataset | List[DataArray]
#             Left components of the fitted model.
#         components2: DataArray | Dataset | List[DataArray]
#             Right components of the fitted model.

#         """
#         return super().components()

#     def scores(self):
#         """Return the scores of the left and right field.

#         The scores in MCA are the projection of the left and right field onto the
#         left and right singular vector of the cross-covariance matrix.

#         Returns
#         -------
#         scores1: DataArray
#             Left scores.
#         scores2: DataArray
#             Right scores.

#         """
#         return super().scores()

#     def homogeneous_patterns(self, correction=None, alpha=0.05):
#         """Return the homogeneous patterns of the left and right field.

#         The homogeneous patterns are the correlation coefficients between the
#         input data and the scores.

#         More precisely, the homogeneous patterns `r_{hom}` are defined as

#         .. math::
#           r_{hom, x} = corr \\left(X, A_x \\right)
#         .. math::
#           r_{hom, y} = corr \\left(Y, A_y \\right)

#         where :math:`X` and :math:`Y` are the input data, :math:`A_x` and :math:`A_y`
#         are the scores of the left and right field, respectively.

#         Parameters
#         ----------
#         correction: str, default=None
#             Method to apply a multiple testing correction. If None, no correction
#             is applied.  Available methods are:
#             - bonferroni : one-step correction
#             - sidak : one-step correction
#             - holm-sidak : step down method using Sidak adjustments
#             - holm : step-down method using Bonferroni adjustments
#             - simes-hochberg : step-up method (independent)
#             - hommel : closed method based on Simes tests (non-negative)
#             - fdr_bh : Benjamini/Hochberg (non-negative) (default)
#             - fdr_by : Benjamini/Yekutieli (negative)
#             - fdr_tsbh : two stage fdr correction (non-negative)
#             - fdr_tsbky : two stage fdr correction (non-negative)
#         alpha: float, default=0.05
#             The desired family-wise error rate. Not used if `correction` is None.

#         Returns
#         -------
#         patterns1: DataArray | Dataset | List[DataArray]
#             Left homogenous patterns.
#         patterns2: DataArray | Dataset | List[DataArray]
#             Right homogenous patterns.
#         pvals1: DataArray | Dataset | List[DataArray]
#             Left p-values.
#         pvals2: DataArray | Dataset | List[DataArray]
#             Right p-values.

#         """
#         input_data1 = self.data["input_data1"]
#         input_data2 = self.data["input_data2"]

#         scores1 = self.data["scores1"]
#         scores2 = self.data["scores2"]

#         hom_pat1, pvals1 = pearson_correlation(
#             input_data1, scores1, correction=correction, alpha=alpha
#         )
#         hom_pat2, pvals2 = pearson_correlation(
#             input_data2, scores2, correction=correction, alpha=alpha
#         )

#         hom_pat1.name = "left_homogeneous_patterns"
#         hom_pat2.name = "right_homogeneous_patterns"

#         pvals1.name = "pvalues_of_left_homogeneous_patterns"
#         pvals2.name = "pvalues_of_right_homogeneous_patterns"

#         hom_pat1 = self.preprocessor1.inverse_transform_components(hom_pat1)
#         hom_pat2 = self.preprocessor2.inverse_transform_components(hom_pat2)

#         pvals1 = self.preprocessor1.inverse_transform_components(pvals1)
#         pvals2 = self.preprocessor2.inverse_transform_components(pvals2)

#         return (hom_pat1, hom_pat2), (pvals1, pvals2)

#     def heterogeneous_patterns(self, correction=None, alpha=0.05):
#         """Return the heterogeneous patterns of the left and right field.

#         The heterogeneous patterns are the correlation coefficients between the
#         input data and the scores of the other field.

#         More precisely, the heterogeneous patterns `r_{het}` are defined as

#         .. math::
#           r_{het, x} = corr \\left(X, A_y \\right)
#         .. math::
#           r_{het, y} = corr \\left(Y, A_x \\right)

#         where :math:`X` and :math:`Y` are the input data, :math:`A_x` and :math:`A_y`
#         are the scores of the left and right field, respectively.

#         Parameters
#         ----------
#         correction: str, default=None
#             Method to apply a multiple testing correction. If None, no correction
#             is applied.  Available methods are:
#             - bonferroni : one-step correction
#             - sidak : one-step correction
#             - holm-sidak : step down method using Sidak adjustments
#             - holm : step-down method using Bonferroni adjustments
#             - simes-hochberg : step-up method (independent)
#             - hommel : closed method based on Simes tests (non-negative)
#             - fdr_bh : Benjamini/Hochberg (non-negative) (default)
#             - fdr_by : Benjamini/Yekutieli (negative)
#             - fdr_tsbh : two stage fdr correction (non-negative)
#             - fdr_tsbky : two stage fdr correction (non-negative)
#         alpha: float, default=0.05
#             The desired family-wise error rate. Not used if `correction` is None.

#         """
#         input_data1 = self.data["input_data1"]
#         input_data2 = self.data["input_data2"]

#         scores1 = self.data["scores1"]
#         scores2 = self.data["scores2"]

#         patterns1, pvals1 = pearson_correlation(
#             input_data1, scores2, correction=correction, alpha=alpha
#         )
#         patterns2, pvals2 = pearson_correlation(
#             input_data2, scores1, correction=correction, alpha=alpha
#         )

#         patterns1.name = "left_heterogeneous_patterns"
#         patterns2.name = "right_heterogeneous_patterns"

#         pvals1.name = "pvalues_of_left_heterogeneous_patterns"
#         pvals2.name = "pvalues_of_right_heterogeneous_patterns"

#         patterns1 = self.preprocessor1.inverse_transform_components(patterns1)
#         patterns2 = self.preprocessor2.inverse_transform_components(patterns2)

#         pvals1 = self.preprocessor1.inverse_transform_components(pvals1)
#         pvals2 = self.preprocessor2.inverse_transform_components(pvals2)

#         return (patterns1, patterns2), (pvals1, pvals2)

#     def _validate_loaded_data(self, data: xr.DataArray):
#         if data.attrs.get("placeholder"):
#             warnings.warn(
#                 f"The input data field '{data.name}' was not saved, which will produce"
#                 " empty results when calling `homogeneous_patterns()` or "
#                 "`heterogeneous_patterns()`. To avoid this warning, you can save the"
#                 " model with `save_data=True`, or add the data manually by running"
#                 " it through the model's `preprocessor.transform()` method and then"
#                 " attaching it with `data.add()`."
#             )


# class ComplexMCA(MCA):
#     """Complex MCA.

#     Complex MCA, also referred to as Analytical SVD (ASVD) by Elipot et al. (2017) [1]_,
#     enhances traditional MCA by accommodating both amplitude and phase information.
#     It achieves this by utilizing the Hilbert transform to preprocess the data,
#     thus allowing for a more comprehensive analysis in the subsequent MCA computation.

#     An optional padding with exponentially decaying values can be applied prior to
#     the Hilbert transform in order to mitigate the impact of spectral leakage.

#     Parameters
#     ----------
#     n_modes: int, default=2
#         Number of modes to calculate.
#     padding : str, optional
#         Specifies the method used for padding the data prior to applying the Hilbert
#         transform. This can help to mitigate the effect of spectral leakage.
#         Currently, only 'exp' for exponential padding is supported. Default is 'exp'.
#     decay_factor : float, optional
#         Specifies the decay factor used in the exponential padding. This parameter
#         is only used if padding='exp'. The recommended value typically ranges between 0.05 to 0.2
#         but ultimately depends on the variability in the data.
#         A smaller value (e.g. 0.05) is recommended for
#         data with high variability, while a larger value (e.g. 0.2) is recommended
#         for data with low variability. Default is 0.2.
#     center: bool, default=True
#         Whether to center the input data.
#     standardize: bool, default=False
#         Whether to standardize the input data.
#     use_coslat: bool, default=False
#         Whether to use cosine of latitude for scaling.
#     n_pca_modes: int, default=None
#         The number of principal components to retain during the PCA preprocessing
#         step applied to both data sets prior to executing MCA.
#         If set to None, PCA preprocessing will be bypassed, and the MCA will be performed on the original datasets.
#         Specifying an integer value greater than 0 for `n_pca_modes` will trigger the PCA preprocessing, retaining
#         only the specified number of principal components. This reduction in dimensionality can be especially beneficial
#         when dealing with high-dimensional data, where computing the cross-covariance matrix can become computationally
#         intensive or in scenarios where multicollinearity is a concern.
#     compute : bool, default=True
#         Whether to compute elements of the model eagerly, or to defer computation.
#         If True, four pieces of the fit will be computed sequentially: 1) the
#         preprocessor scaler, 2) optional NaN checks, 3) SVD decomposition, 4) scores
#         and components.
#     sample_name: str, default="sample"
#         Name of the new sample dimension.
#     feature_name: str, default="feature"
#         Name of the new feature dimension.
#     solver: {"auto", "full", "randomized"}, default="auto"
#         Solver to use for the SVD computation.
#     random_state: int, optional
#         Random state for randomized SVD solver.
#     solver_kwargs: dict, default={}
#         Additional keyword arguments passed to the SVD solver.

#     Notes
#     -----
#     Complex MCA extends MCA to complex-valued data that contain both magnitude and phase information.
#     The Hilbert transform is used to transform real-valued data to complex-valued data, from which both
#     amplitude and phase can be extracted.

#     Similar to MCA, Complex MCA is used in climate science to identify coupled patterns of variability
#     between two different climate variables. But unlike MCA, Complex MCA can identify coupled patterns
#     that involve phase shifts.

#     References
#     ----------
#     .. [1] Elipot, S., Frajka-Williams, E., Hughes, C. W., Olhede, S. & Lankhorst, M. Observed Basin-Scale Response of the North Atlantic Meridional Overturning Circulation to Wind Stress Forcing. Journal of Climate 30, 2029–2054 (2017).


#     Examples
#     --------
#     >>> model = ComplexMCA(n_modes=5, standardize=True)
#     >>> model.fit(data1, data2)

#     """

#     def __init__(
#         self,
#         n_modes: int = 2,
#         padding: str = "exp",
#         decay_factor: float = 0.2,
#         center: bool = True,
#         standardize: bool = False,
#         use_coslat: bool = False,
#         check_nans: bool = True,
#         n_pca_modes: Optional[int] = None,
#         compute: bool = True,
#         sample_name: str = "sample",
#         feature_name: str = "feature",
#         solver: str = "auto",
#         random_state: Optional[bool] = None,
#         solver_kwargs: dict = {},
#         **kwargs,
#     ):
#         super().__init__(
#             n_modes=n_modes,
#             center=center,
#             standardize=standardize,
#             use_coslat=use_coslat,
#             check_nans=check_nans,
#             n_pca_modes=n_pca_modes,
#             compute=compute,
#             sample_name=sample_name,
#             feature_name=feature_name,
#             solver=solver,
#             random_state=random_state,
#             solver_kwargs=solver_kwargs,
#             **kwargs,
#         )
#         self.attrs.update({"model": "Complex MCA"})
#         self._params.update({"padding": padding, "decay_factor": decay_factor})

#     def _fit_algorithm(self, data1: DataArray, data2: DataArray) -> Self:
#         assert_not_complex(data1)
#         assert_not_complex(data2)

#         sample_name = self.sample_name
#         feature_name = self.feature_name

#         # Settings for Hilbert transform
#         hilbert_kwargs = {
#             "padding": self._params["padding"],
#             "decay_factor": self._params["decay_factor"],
#         }

#         # Initialize the SVD decomposer
#         decomposer = Decomposer(**self._decomposer_kwargs)

#         # Perform SVD on PCA-reduced data
#         if (self.pca1 is not None) and (self.pca2 is not None):
#             # Fit the PCA models
#             self.pca1.fit(data1, sample_name)
#             self.pca2.fit(data2, sample_name)
#             # Get the PCA scores
#             pca_scores1 = self.pca1.data["scores"] * self.pca1.singular_values()
#             pca_scores2 = self.pca2.data["scores"] * self.pca2.singular_values()
#             # Apply hilbert transform
#             pca_scores1 = hilbert_transform(
#                 pca_scores1, dims=(sample_name, "mode"), **hilbert_kwargs
#             )
#             pca_scores2 = hilbert_transform(
#                 pca_scores2, dims=(sample_name, "mode"), **hilbert_kwargs
#             )
#             # Compute the cross-covariance matrix of the PCA scores
#             pca_scores1 = pca_scores1.rename({"mode": "feature_temp1"})
#             pca_scores2 = pca_scores2.rename({"mode": "feature_temp2"})
#             cov_matrix = self._compute_cross_covariance_matrix(pca_scores1, pca_scores2)

#             # Perform the SVD
#             decomposer.fit(cov_matrix, dims=("feature_temp1", "feature_temp2"))
#             V1 = decomposer.U_  # left singular vectors (feature_temp1 x mode)
#             V2 = decomposer.V_  # right singular vectors (feature_temp2 x mode)

#             # left and right PCA eigenvectors (feature_name x mode)
#             V1pre = self.pca1.data["components"]
#             V2pre = self.pca2.data["components"]

#             # Compute the singular vectors
#             V1pre = V1pre.rename({"mode": "feature_temp1"})
#             V2pre = V2pre.rename({"mode": "feature_temp2"})
#             singular_vectors1 = xr.dot(V1pre, V1, dims="feature_temp1")
#             singular_vectors2 = xr.dot(V2pre, V2, dims="feature_temp2")

#         # Perform SVD directly on data
#         else:
#             # Perform Hilbert transform
#             data1 = hilbert_transform(
#                 data1, dims=(sample_name, feature_name), **hilbert_kwargs
#             )
#             data2 = hilbert_transform(
#                 data2, dims=(sample_name, feature_name), **hilbert_kwargs
#             )
#             # Rename feature and associated dimensions of data objects to avoid index conflicts
#             dim_renamer1 = DimensionRenamer(feature_name, "1")
#             dim_renamer2 = DimensionRenamer(feature_name, "2")
#             data1_temp = dim_renamer1.fit_transform(data1)
#             data2_temp = dim_renamer2.fit_transform(data2)
#             # Compute the cross-covariance matrix
#             cov_matrix = self._compute_cross_covariance_matrix(data1_temp, data2_temp)

#             # Perform the SVD
#             decomposer.fit(cov_matrix, dims=("feature1", "feature2"))
#             singular_vectors1 = decomposer.U_
#             singular_vectors2 = decomposer.V_

#             # Rename the singular vectors
#             singular_vectors1 = dim_renamer1.inverse_transform(singular_vectors1)
#             singular_vectors2 = dim_renamer2.inverse_transform(singular_vectors2)

#         # Store the results
#         singular_values = decomposer.s_

#         # Compute total squared variance
#         squared_covariance = singular_values**2
#         total_squared_covariance = (abs(cov_matrix) ** 2).sum()

#         norm1 = np.sqrt(singular_values)
#         norm2 = np.sqrt(singular_values)

#         # Index of the sorted squared covariance
#         idx_sorted_modes = argsort_dask(squared_covariance, "mode")[::-1]
#         idx_sorted_modes.coords.update(squared_covariance.coords)

#         # Project the data onto the singular vectors
#         scores1 = xr.dot(data1, singular_vectors1) / norm1
#         scores2 = xr.dot(data2, singular_vectors2) / norm2

#         self.data.add(name="input_data1", data=data1, allow_compute=False)
#         self.data.add(name="input_data2", data=data2, allow_compute=False)
#         self.data.add(name="components1", data=singular_vectors1)
#         self.data.add(name="components2", data=singular_vectors2)
#         self.data.add(name="scores1", data=scores1)
#         self.data.add(name="scores2", data=scores2)
#         self.data.add(name="squared_covariance", data=squared_covariance)
#         self.data.add(name="total_squared_covariance", data=total_squared_covariance)
#         self.data.add(name="idx_modes_sorted", data=idx_sorted_modes)
#         self.data.add(name="norm1", data=norm1)
#         self.data.add(name="norm2", data=norm2)

#         # Assign analysis relevant meta data
#         self.data.set_attrs(self.attrs)
#         return self

#     def components_amplitude(self) -> Tuple[DataObject, DataObject]:
#         """Compute the amplitude of the components.

#         The amplitude of the components are defined as

#         .. math::
#             A_ij = |C_ij|

#         where :math:`C_{ij}` is the :math:`i`-th entry of the :math:`j`-th component and
#         :math:`|\\cdot|` denotes the absolute value.

#         Returns
#         -------
#         DataObject
#             Amplitude of the left components.
#         DataObject
#             Amplitude of the left components.

#         """
#         comps1 = abs(self.data["components1"])
#         comps1.name = "left_components_amplitude"

#         comps2 = abs(self.data["components2"])
#         comps2.name = "right_components_amplitude"

#         comps1 = self.preprocessor1.inverse_transform_components(comps1)
#         comps2 = self.preprocessor2.inverse_transform_components(comps2)

#         return (comps1, comps2)

#     def components_phase(self) -> Tuple[DataObject, DataObject]:
#         """Compute the phase of the components.

#         The phase of the components are defined as

#         .. math::
#             \\phi_{ij} = \\arg(C_{ij})

#         where :math:`C_{ij}` is the :math:`i`-th entry of the :math:`j`-th component and
#         :math:`\\arg(\\cdot)` denotes the argument of a complex number.

#         Returns
#         -------
#         DataObject
#             Phase of the left components.
#         DataObject
#             Phase of the right components.

#         """
#         comps1 = xr.apply_ufunc(np.angle, self.data["components1"], keep_attrs=True)
#         comps1.name = "left_components_phase"

#         comps2 = xr.apply_ufunc(np.angle, self.data["components2"], keep_attrs=True)
#         comps2.name = "right_components_phase"

#         comps1 = self.preprocessor1.inverse_transform_components(comps1)
#         comps2 = self.preprocessor2.inverse_transform_components(comps2)

#         return (comps1, comps2)

#     def scores_amplitude(self) -> Tuple[DataArray, DataArray]:
#         """Compute the amplitude of the scores.

#         The amplitude of the scores are defined as

#         .. math::
#             A_ij = |S_ij|

#         where :math:`S_{ij}` is the :math:`i`-th entry of the :math:`j`-th score and
#         :math:`|\\cdot|` denotes the absolute value.

#         Returns
#         -------
#         DataArray
#             Amplitude of the left scores.
#         DataArray
#             Amplitude of the right scores.

#         """
#         scores1 = abs(self.data["scores1"])
#         scores2 = abs(self.data["scores2"])

#         scores1.name = "left_scores_amplitude"
#         scores2.name = "right_scores_amplitude"

#         scores1 = self.preprocessor1.inverse_transform_scores(scores1)
#         scores2 = self.preprocessor2.inverse_transform_scores(scores2)
#         return (scores1, scores2)

#     def scores_phase(self) -> Tuple[DataArray, DataArray]:
#         """Compute the phase of the scores.

#         The phase of the scores are defined as

#         .. math::
#             \\phi_{ij} = \\arg(S_{ij})

#         where :math:`S_{ij}` is the :math:`i`-th entry of the :math:`j`-th score and
#         :math:`\\arg(\\cdot)` denotes the argument of a complex number.

#         Returns
#         -------
#         DataArray
#             Phase of the left scores.
#         DataArray
#             Phase of the right scores.

#         """
#         scores1 = xr.apply_ufunc(np.angle, self.data["scores1"], keep_attrs=True)
#         scores2 = xr.apply_ufunc(np.angle, self.data["scores2"], keep_attrs=True)

#         scores1.name = "left_scores_phase"
#         scores2.name = "right_scores_phase"

#         scores1 = self.preprocessor1.inverse_transform_scores(scores1)
#         scores2 = self.preprocessor2.inverse_transform_scores(scores2)

#         return (scores1, scores2)

#     def transform(self, data1: DataObject, data2: DataObject):
#         raise NotImplementedError("Complex MCA does not support transform method.")

#     def _inverse_transform_algorithm(self, scores1: DataArray, scores2: DataArray):
#         data1, data2 = super()._inverse_transform_algorithm(scores1, scores2)

#         # Enforce real output
#         return data1.real, data2.real

#     def homogeneous_patterns(self, correction=None, alpha=0.05):
#         raise NotImplementedError(
#             "Complex MCA does not support homogeneous_patterns method."
#         )

#     def heterogeneous_patterns(self, correction=None, alpha=0.05):
#         raise NotImplementedError(
#             "Complex MCA does not support heterogeneous_patterns method."
#         )
