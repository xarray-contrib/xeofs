import warnings
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from typing_extensions import Self

from ..utils.data_types import DataArray, DataObject
from ..utils.hilbert_transform import hilbert_transform
from ..utils.linalg import fractional_matrix_power

# from ..utils.sanity_checks import assert_not_complex
from ..utils.statistics import pearson_correlation
from ..utils.xarray_utils import argsort_dask
from ._base_model_cross_set import _BaseModelCrossSet
from .decomposer import Decomposer


class ContinuousPowerCCA(_BaseModelCrossSet):
    """Continuous Power CCA.

    CCA is a statistical method that finds patterns of maximum correlation between two datasets.

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

    Notes
    -----
    MCA is similar to Principal Component Analysis (PCA) and Canonical Correlation Analysis (CCA),
    but while PCA finds modes of maximum variance and CCA finds modes of maximum correlation,
    MCA finds modes of maximum covariance. See [1]_ [2]_ for more details.

    References
    ----------
    .. [1] Bretherton, C., Smith, C., Wallace, J., 1992. An intercomparison of methods for finding coupled patterns in climate data. Journal of climate 5, 541–560.
    .. [2] Cherry, S., 1996. Singular value decomposition analysis and canonical correlation analysis. Journal of Climate 9, 2003–2009.

    Examples
    --------
    >>> model = MCA(n_modes=5, standardize=True)
    >>> model.fit(data1, data2)

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
        alpha: Sequence[float] | float = 1.0,
        compute: bool = True,
        sample_name: str = "sample",
        feature_name: Sequence[str] | str = "feature",
        solver: str = "auto",
        random_state: Optional[int] = None,
        solver_kwargs: Dict = {},
    ):
        super().__init__(
            n_modes=n_modes,
            center=True,
            standardize=standardize,
            use_coslat=use_coslat,
            check_nans=check_nans,
            use_pca=use_pca,
            n_pca_modes=n_pca_modes,
            pca_init_rank_reduction=pca_init_rank_reduction,
            alpha=alpha,
            compute=compute,
            sample_name=sample_name,
            feature_name=feature_name,
            solver=solver,
            random_state=random_state,
            solver_kwargs=solver_kwargs,
        )
        self.attrs.update({"model": "Continuous Power CCA"})

        params = self.get_params()
        self.sample_name: str = params["sample_name"]
        self.feature_name: tuple[str, str] = params["feature_name"]

    def _compute_cross_covariance_matrix(
        self, X1: DataArray, X2: DataArray
    ) -> DataArray:
        """Compute the cross-covariance matrix of two data objects.

        Note: Assume that the data objects are centered.

        """

        def _compute_cross_covariance_numpy(X1, X2):
            n_samples = X1.shape[0]
            return X1.conj().T @ X2 / (n_samples - 1)

        sample_name = self.sample_name
        n_samples1 = X1.coords[sample_name].size
        n_samples2 = X2.coords[sample_name].size
        if n_samples1 != n_samples2:
            err_msg = f"Both data matrices must have the same number of samples but found {n_samples1} in the first and {n_samples2} in the second."
            raise ValueError(err_msg)

        if X1.dims[1] == X2.dims[1]:
            raise ValueError(
                "The two data matrices must have different feature dimensions, set `feature_name` to different values for each data matrix."
            )

        # Rename the sample dimension to avoid conflicts for
        # different coordinates with same length
        sample_name_x = "sample_dim_x"
        sample_name_y = "sample_dim_y"
        X1 = X1.rename({self.sample_name: sample_name_x})
        X2 = X2.rename({self.sample_name: sample_name_y})
        return xr.apply_ufunc(
            _compute_cross_covariance_numpy,
            X1,
            X2,
            input_core_dims=[
                [sample_name_x, self.feature_name[0]],
                [sample_name_y, self.feature_name[1]],
            ],
            output_core_dims=[[self.feature_name[0], self.feature_name[1]]],
            dask="allowed",
        )

    def _compute_total_squared_covariance(self, C: DataArray) -> DataArray:
        """Compute the total squared covariance.

        Requires the unwhitened covariance matrix which we can obtain by multiplying the whitened covariance matrix with the inverse of the whitening transformation matrix.
        """
        C = self.whitener2.inverse_transform_data(C)
        C = self.whitener1.inverse_transform_data(C.conj().T)
        # Not necessary to conjugate transpose for total squared covariance
        # C = C.conj().T
        return (abs(C) ** 2).sum()

    def _fit_algorithm(
        self,
        data1: DataArray,
        data2: DataArray,
    ) -> Self:
        feature_name = self.feature_name

        # Compute the totalsquared covariance from the unwhitened data
        C_whitened = self._compute_cross_covariance_matrix(data1, data2)

        # Initialize the SVD decomposer
        decomposer = Decomposer(**self._decomposer_kwargs)
        dims = (feature_name[0], feature_name[1])
        decomposer.fit(C_whitened, dims=dims)

        # Store the results
        singular_values = decomposer.s_
        Q1 = decomposer.U_
        Q2 = decomposer.V_

        # Compute total squared variance
        total_squared_covariance = self._compute_total_squared_covariance(C_whitened)

        # Index of the sorted squared covariance
        idx_sorted_modes = argsort_dask(singular_values, "mode")[::-1]
        idx_sorted_modes.coords.update(singular_values.coords)

        # Project the data onto the singular vectors
        scores1 = xr.dot(data1, Q1, dims=feature_name[0])
        scores2 = xr.dot(data2, Q2, dims=feature_name[1])

        norm1 = np.sqrt(xr.dot(scores1.conj(), scores1, dims=self.sample_name)).real
        norm2 = np.sqrt(xr.dot(scores2.conj(), scores2, dims=self.sample_name)).real

        self.data.add(name="input_data1", data=data1, allow_compute=False)
        self.data.add(name="input_data2", data=data2, allow_compute=False)
        self.data.add(name="components1", data=Q1)
        self.data.add(name="components2", data=Q2)
        self.data.add(name="scores1", data=scores1)
        self.data.add(name="scores2", data=scores2)
        self.data.add(name="singular_values", data=singular_values)
        self.data.add(name="total_squared_covariance", data=total_squared_covariance)
        self.data.add(name="idx_modes_sorted", data=idx_sorted_modes)
        self.data.add(name="norm1", data=norm1)
        self.data.add(name="norm2", data=norm2)

        # # Assign analysis-relevant meta data
        self.data.set_attrs(self.attrs)
        return self

    def _get_scores(self, normalized=False):
        norm1 = self.data["norm1"]
        norm2 = self.data["norm2"]

        scores1 = self.data["scores1"]
        scores2 = self.data["scores2"]

        if normalized:
            scores1 = scores1 / norm1
            scores2 = scores2 / norm2

        return scores1, scores2

    def _get_components(self, normalized=True):
        comps1 = self.data["components1"]
        comps2 = self.data["components2"]

        if not normalized:
            comps1 = comps1 * self.data["norm1"]
            comps2 = comps2 * self.data["norm2"]

        return comps1, comps2

    def _transform_algorithm(
        self,
        data1: Optional[DataArray] = None,
        data2: Optional[DataArray] = None,
        normalized=False,
    ) -> Dict[str, DataArray]:
        results = {}
        if data1 is not None:
            # Project data onto singular vectors
            comps1 = self.data["components1"]
            norm1 = self.data["norm1"]
            scores1 = xr.dot(data1, comps1)
            if normalized:
                scores1 = scores1 / norm1
            results["data1"] = scores1

        if data2 is not None:
            # Project data onto singular vectors
            comps2 = self.data["components2"]
            norm2 = self.data["norm2"]
            scores2 = xr.dot(data2, comps2)
            if normalized:
                scores2 = scores2 / norm2
            results["data2"] = scores2

        return results

    def _inverse_transform_algorithm(
        self, scores1: DataArray, scores2: DataArray
    ) -> Tuple[DataArray, DataArray]:
        """Reconstruct the original data from transformed data.

        Parameters
        ----------
        scores1: DataArray
            Transformed left field data to be reconstructed. This could be
            a subset of the `scores` data of a fitted model, or unseen data.
            Must have a 'mode' dimension.
        scores2: DataArray
            Transformed right field data to be reconstructed. This could be
            a subset of the `scores` data of a fitted model, or unseen data.
            Must have a 'mode' dimension.

        Returns
        -------
        Xrec1: DataArray
            Reconstructed data of left field.
        Xrec2: DataArray
            Reconstructed data of right field.

        """
        # Singular vectors
        comps1 = self.data["components1"].sel(mode=scores1.mode)
        comps2 = self.data["components2"].sel(mode=scores2.mode)

        # Reconstruct the data
        data1 = xr.dot(scores1, comps1.conj(), dims="mode")
        data2 = xr.dot(scores2, comps2.conj(), dims="mode")

        return data1, data2

    def _predict_algorithm(self, X: DataArray) -> DataArray:
        sample_name_fit_x = "sample_fit_dim_x"
        sample_name_fit_y = "sample_fit_dim_y"
        Qx = self.data["components1"]
        Rx = self.data["scores1"].rename({self.sample_name: sample_name_fit_x})
        Ry = self.data["scores2"].rename({self.sample_name: sample_name_fit_y})

        def _predict_numpy(X, Qx, Rx, Ry):
            G = Rx.conj().T @ Ry / np.linalg.norm(Rx, axis=0) ** 2
            return X @ Qx @ G

        Ry_pred = xr.apply_ufunc(
            _predict_numpy,
            X,
            Qx,
            Rx,
            Ry,
            input_core_dims=[
                [self.sample_name, self.feature_name[0]],
                [self.feature_name[0], "mode"],
                [sample_name_fit_x, "mode"],
                [sample_name_fit_y, "mode"],
            ],
            output_core_dims=[[self.sample_name, "mode"]],
            dask="allowed",
        )
        Ry_pred.name = "pseudo_scores_Y"
        return Ry_pred

    def singular_values(self):
        """Get the singular values of the cross-covariance matrix."""
        singular_values = self.data["singular_values"]
        singular_values.name = "singular_values"
        return singular_values

    def _squared_covariance(self):
        """Get the squared covariance.

        The squared covariance corresponds to the explained variance in PCA and is given by the
        squared singular values of the covariance matrix.

        """
        return self.data["singular_values"] ** 2

    def squared_covariance_fraction(self):
        """Calculate the squared covariance fraction (SCF).

        The SCF is a measure of the proportion of the total squared covariance that is explained by each mode `i`. It is computed
        as follows:

        .. math::
            SCF_i = \\frac{\\sigma_i^2}{\\sum_{i=1}^{m} \\sigma_i^2}

        where `m` is the total number of modes and :math:`\\sigma_i` is the `ith` singular value of the covariance matrix.

        """
        return self._squared_covariance() / self.data["total_squared_covariance"]

    def cross_correlation_coefficients(self):
        """Get the cross-correlation coefficients.

        The cross-correlation coefficients are the correlation coefficients between the left and right scores.

        """

        def _compute_cross_corr_numpy(Rx, Ry):
            return np.diag(Rx.conj().T @ Ry)

        Rx = self.data["scores1"] / self.data["norm1"]
        Ry = self.data["scores2"] / self.data["norm2"]

        # Rename the sample dimension to avoid conflicts for
        # different coordinates with same length
        sample_name_x = "sample_dim_x"
        sample_name_y = "sample_dim_y"
        Rx = Rx.rename({self.sample_name: sample_name_x})
        Ry = Ry.rename({self.sample_name: sample_name_y})

        cross_corr = xr.apply_ufunc(
            _compute_cross_corr_numpy,
            Rx,
            Ry,
            input_core_dims=[[sample_name_x, "mode"], [sample_name_y, "mode"]],
            output_core_dims=[["mode"]],
            dask="allowed",
        )
        cross_corr = cross_corr.real
        cross_corr.name = "cross_correlation_coefficients"
        return cross_corr

    def correlation_coefficients_X(self):
        """Get the correlation coefficients of the left scores."""
        Rx = self.data["scores1"] / self.data["norm1"]

        corr = xr.dot(
            Rx.rename({"mode": "mode_i"}).conj(),
            Rx.rename({"mode": "mode_j"}),
            dims=self.sample_name,
        )
        corr.name = "correlation_coefficients_X"
        return corr

    def correlation_coefficients_Y(self):
        """Get the correlation coefficients of the right scores."""
        Ry = self.data["scores2"] / self.data["norm2"]

        corr = xr.dot(
            Ry.rename({"mode": "mode_i"}).conj(),
            Ry.rename({"mode": "mode_j"}),
            dims=self.sample_name,
        )
        corr.name = "correlation_coefficients_Y"
        return corr

    def _total_covariance(self) -> DataArray:
        """Get the total covariance.

        This measure follows the defintion of Cheng and Dunkerton (1995).
        Note that this measure is not an invariant in MCA.

        """
        tot_cov = self.singular_values().sum()
        tot_cov.attrs.update(self.singular_values().attrs)
        tot_cov.name = "total_covariance"
        return tot_cov

    def covariance_fraction(self):
        """Get the covariance fraction (CF).

        Cheng and Dunkerton (1995) define the CF as follows:

        .. math::
            CF_i = \\frac{\\sigma_i}{\\sum_{i=1}^{m} \\sigma_i}

        where `m` is the total number of modes and :math:`\\sigma_i` is the
        `ith` singular value of the covariance matrix.

        In this implementation the sum of singular values is estimated from
        the first `n` modes, therefore one should aim to retain as many
        modes as possible to get a good estimate of the covariance fraction.

        Note
        ----
        It is important to differentiate the CF from the squared covariance fraction (SCF). While the SCF is an invariant quantity in MCA, the CF is not.
        Therefore, the SCF is used to assess the relative importance of each mode. Cheng and Dunkerton (1995) introduced the CF in the context of
        Varimax-rotated MCA to compare the relative importance of each mode before and after rotation. In the special case of both data fields in MCA being identical,
        the CF is equivalent to the explained variance ratio in PCA.

        """
        # Check how sensitive the CF is to the number of modes
        svals = self.singular_values()
        tot_var = self._total_covariance()
        cf = svals[0] / svals.cumsum()
        change_per_mode = cf.shift({"mode": 1}) - cf
        change_in_cf_in_last_mode = change_per_mode.isel(mode=-1)
        if change_in_cf_in_last_mode > 0.001:
            print(
                "Warning: CF is sensitive to the number of modes retained. Please increase `n_modes` for a better estimate."
            )
        cov_frac = svals / tot_var
        cov_frac.name = "covariance_fraction"
        cov_frac.attrs.update(svals.attrs)
        return cov_frac

    def _compute_total_variance(self, X: DataArray) -> DataArray:
        """Compute the total variance of the input data."""
        return (abs(X) ** 2).sum()

    def fraction_variance_X_explained_by_X(self):
        """Compute the fraction of variance explained in the left field explained by the left scores.

        This is based on the equation 14.46 a in [1]_.

        References
        ----------
        .. [1] Wilks, D. S. Statistical Methods in the Atmospheric Sciences. (Academic Press, 2019). doi:https://doi.org/10.1016/B978-0-12-815823-4.00011-0.

        """
        scores1 = self.data["scores1"]
        Sxx = xr.dot(scores1.conj(), scores1, dims=self.sample_name)

        # Compute total variance of X1
        X1 = self.data["input_data1"]
        # Unwhiten the data
        X1 = self.whitener1.inverse_transform_data(X1, unwhiten_only=True)
        total_variance_X = self._compute_total_variance(X1)

        fve_x = Sxx / total_variance_X
        fve_x.name = "fraction_variance_X_explained_by_X"
        return fve_x

    def fraction_variance_Y_explained_by_Y(self):
        """Compute the fraction of variance explained in the right field explained by the right scores.

        This is based on the equation 14.46 b in [1]_.

        References
        ----------
        .. [1] Wilks, D. S. Statistical Methods in the Atmospheric Sciences. (Academic Press, 2019). doi:https://doi.org/10.1016/B978-0-12-815823-4.00011-0.

        """
        scores2 = self.data["scores2"]
        Syy = xr.dot(scores2.conj(), scores2, dims=self.sample_name)

        # Compute total variance of X1
        X2 = self.data["input_data2"]
        # Unwhiten the data
        X2 = self.whitener2.inverse_transform_data(X2, unwhiten_only=True)
        total_variance_Y = self._compute_total_variance(X2)

        fve_y = Syy / total_variance_Y
        fve_y.name = "fraction_variance_Y_explained_by_Y"
        return fve_y

    def fraction_variance_Y_explained_by_X(self) -> DataArray:
        """Get the fraction of variance in the right field explained by the left scores.

        This is based on equation 16 in [1]_.

        References
        ----------
        .. [1] Swenson, E. Continuum Power CCA: A Unified Approach for Isolating Coupled Modes. Journal of Climate 28, 1016–1030 (2015).

        """

        def _compute_total_variance_numpy(X, Y):
            Tinv = fractional_matrix_power(X.conj().T @ X, -0.5)
            return np.linalg.norm(Tinv @ X.conj().T @ Y / (X.shape[0] - 1)) ** 2

        def _compute_residual_variance_numpy(X, Y, Xrec, Yrec):
            dX = X - Xrec
            dY = Y - Yrec

            Tinv = fractional_matrix_power(X.conj().T @ X, -0.5)
            return np.linalg.norm(Tinv @ dX.conj().T @ dY / (dX.shape[0] - 1)) ** 2

        sample_name_x = "sample_dim_x"
        sample_name_y = "sample_dim_y"

        # Get the singular vectors
        Q1 = self.data["components1"]
        Q2 = self.data["components2"]

        # Get input data
        X1 = self.data["input_data1"]
        X2 = self.data["input_data2"]

        # Unwhiten the data
        X1 = self.whitener1.inverse_transform_data(X1, unwhiten_only=True)
        X2 = self.whitener2.inverse_transform_data(X2, unwhiten_only=True)

        # Compute the total variance
        X1 = X1.rename({self.sample_name: sample_name_x})
        X2 = X2.rename({self.sample_name: sample_name_y})
        total_variance: DataArray = xr.apply_ufunc(
            _compute_total_variance_numpy,
            X1,
            X2,
            input_core_dims=[
                [sample_name_x, self.feature_name[0]],
                [sample_name_y, self.feature_name[1]],
            ],
            output_core_dims=[[]],
            dask="allowed",
        )

        # Get the component scores
        scores1 = self.data["scores1"]
        scores2 = self.data["scores2"]

        # Compute the residual variance for each mode
        fraction_variance_explained: list[DataArray] = []
        for mode in scores1.mode.values:
            # Reconstruct the data
            X1r = xr.dot(
                scores1.sel(mode=[mode]), Q1.sel(mode=[mode]).conj().T, dims="mode"
            )
            X2r = xr.dot(
                scores2.sel(mode=[mode]), Q2.sel(mode=[mode]).conj().T, dims="mode"
            )

            # Unwhitend the reconstructed data
            X1r = self.whitener1.inverse_transform_data(X1r, unwhiten_only=True)
            X2r = self.whitener2.inverse_transform_data(X2r, unwhiten_only=True)

            # Compute fraction variance explained
            X1r = X1r.rename({self.sample_name: sample_name_x})
            X2r = X2r.rename({self.sample_name: sample_name_y})
            res_var: DataArray = xr.apply_ufunc(
                _compute_residual_variance_numpy,
                X1,
                X2,
                X1r,
                X2r,
                input_core_dims=[
                    [sample_name_x, self.feature_name[0]],
                    [sample_name_y, self.feature_name[1]],
                    [sample_name_x, self.feature_name[0]],
                    [sample_name_y, self.feature_name[1]],
                ],
                output_core_dims=[[]],
                dask="allowed",
            )
            res_var = res_var.expand_dims({"mode": [mode]})
            fraction_variance_explained.append(1 - res_var / total_variance)

        fve_yx = xr.concat(fraction_variance_explained, dim="mode")
        fve_yx.name = "fraction_variance_Y_explained_by_X"
        return fve_yx

    def homogeneous_patterns(self, correction=None, alpha=0.05):
        """Return the homogeneous patterns of the left and right field.

        The homogeneous patterns are the correlation coefficients between the
        input data and the scores.

        More precisely, the homogeneous patterns `r_{hom}` are defined as

        .. math::
          r_{hom, x} = corr \\left(X, A_x \\right)
        .. math::
          r_{hom, y} = corr \\left(Y, A_y \\right)

        where :math:`X` and :math:`Y` are the input data, :math:`A_x` and :math:`A_y`
        are the scores of the left and right field, respectively.

        Parameters
        ----------
        correction: str, default=None
            Method to apply a multiple testing correction. If None, no correction
            is applied.  Available methods are:
            - bonferroni : one-step correction
            - sidak : one-step correction
            - holm-sidak : step down method using Sidak adjustments
            - holm : step-down method using Bonferroni adjustments
            - simes-hochberg : step-up method (independent)
            - hommel : closed method based on Simes tests (non-negative)
            - fdr_bh : Benjamini/Hochberg (non-negative) (default)
            - fdr_by : Benjamini/Yekutieli (negative)
            - fdr_tsbh : two stage fdr correction (non-negative)
            - fdr_tsbky : two stage fdr correction (non-negative)
        alpha: float, default=0.05
            The desired family-wise error rate. Not used if `correction` is None.

        Returns
        -------
        patterns1: DataArray | Dataset | List[DataArray]
            Left homogenous patterns.
        patterns2: DataArray | Dataset | List[DataArray]
            Right homogenous patterns.
        pvals1: DataArray | Dataset | List[DataArray]
            Left p-values.
        pvals2: DataArray | Dataset | List[DataArray]
            Right p-values.

        """
        input_data1 = self.data["input_data1"]
        input_data2 = self.data["input_data2"]

        input_data1 = self.whitener1.inverse_transform_data(input_data1)
        input_data2 = self.whitener2.inverse_transform_data(input_data2)

        scores1 = self.data["scores1"]
        scores2 = self.data["scores2"]

        hom_pat1, pvals1 = pearson_correlation(
            input_data1,
            scores1,
            correction=correction,
            alpha=alpha,
            sample_name=self.sample_name,
            feature_name=self.feature_name[0],
        )
        hom_pat2, pvals2 = pearson_correlation(
            input_data2,
            scores2,
            correction=correction,
            alpha=alpha,
            sample_name=self.sample_name,
            feature_name=self.feature_name[1],
        )

        hom_pat1.name = "left_homogeneous_patterns"
        hom_pat2.name = "right_homogeneous_patterns"

        pvals1.name = "pvalues_of_left_homogeneous_patterns"
        pvals2.name = "pvalues_of_right_homogeneous_patterns"

        hom_pat1 = self.preprocessor1.inverse_transform_components(hom_pat1)
        hom_pat2 = self.preprocessor2.inverse_transform_components(hom_pat2)

        pvals1 = self.preprocessor1.inverse_transform_components(pvals1)
        pvals2 = self.preprocessor2.inverse_transform_components(pvals2)

        return (hom_pat1, hom_pat2), (pvals1, pvals2)

    def heterogeneous_patterns(self, correction=None, alpha=0.05):
        """Return the heterogeneous patterns of the left and right field.

        The heterogeneous patterns are the correlation coefficients between the
        input data and the scores of the other field.

        More precisely, the heterogeneous patterns `r_{het}` are defined as

        .. math::
          r_{het, x} = corr \\left(X, A_y \\right)
        .. math::
          r_{het, y} = corr \\left(Y, A_x \\right)

        where :math:`X` and :math:`Y` are the input data, :math:`A_x` and :math:`A_y`
        are the scores of the left and right field, respectively.

        Parameters
        ----------
        correction: str, default=None
            Method to apply a multiple testing correction. If None, no correction
            is applied.  Available methods are:
            - bonferroni : one-step correction
            - sidak : one-step correction
            - holm-sidak : step down method using Sidak adjustments
            - holm : step-down method using Bonferroni adjustments
            - simes-hochberg : step-up method (independent)
            - hommel : closed method based on Simes tests (non-negative)
            - fdr_bh : Benjamini/Hochberg (non-negative) (default)
            - fdr_by : Benjamini/Yekutieli (negative)
            - fdr_tsbh : two stage fdr correction (non-negative)
            - fdr_tsbky : two stage fdr correction (non-negative)
        alpha: float, default=0.05
            The desired family-wise error rate. Not used if `correction` is None.

        """
        input_data1 = self.data["input_data1"]
        input_data2 = self.data["input_data2"]

        input_data1 = self.whitener1.inverse_transform_data(input_data1)
        input_data2 = self.whitener2.inverse_transform_data(input_data2)

        scores1 = self.data["scores1"]
        scores2 = self.data["scores2"]

        patterns1, pvals1 = pearson_correlation(
            input_data1,
            scores2,
            correction=correction,
            alpha=alpha,
            sample_name=self.sample_name,
            feature_name=self.feature_name[0],
        )
        patterns2, pvals2 = pearson_correlation(
            input_data2,
            scores1,
            correction=correction,
            alpha=alpha,
            sample_name=self.sample_name,
            feature_name=self.feature_name[0],
        )

        patterns1.name = "left_heterogeneous_patterns"
        patterns2.name = "right_heterogeneous_patterns"

        pvals1.name = "pvalues_of_left_heterogeneous_patterns"
        pvals2.name = "pvalues_of_right_heterogeneous_patterns"

        patterns1 = self.preprocessor1.inverse_transform_components(patterns1)
        patterns2 = self.preprocessor2.inverse_transform_components(patterns2)

        pvals1 = self.preprocessor1.inverse_transform_components(pvals1)
        pvals2 = self.preprocessor2.inverse_transform_components(pvals2)

        return (patterns1, patterns2), (pvals1, pvals2)

    def _validate_loaded_data(self, data: xr.DataArray):
        if data.attrs.get("placeholder"):
            warnings.warn(
                f"The input data field '{data.name}' was not saved, which will produce"
                " empty results when calling `homogeneous_patterns()` or "
                "`heterogeneous_patterns()`. To avoid this warning, you can save the"
                " model with `save_data=True`, or add the data manually by running"
                " it through the model's `preprocessor.transform()` method and then"
                " attaching it with `data.add()`."
            )


class ComplexCPCCA(ContinuousPowerCCA):
    """Complex Continuous Power CCA."""

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
        alpha: Sequence[float] | float = 1.0,
        compute: bool = True,
        sample_name: str = "sample",
        feature_name: Sequence[str] | str = "feature",
        solver: str = "auto",
        random_state: Optional[int] = None,
        solver_kwargs: Dict = {},
    ):
        super().__init__(
            n_modes=n_modes,
            standardize=standardize,
            use_coslat=use_coslat,
            check_nans=check_nans,
            use_pca=use_pca,
            n_pca_modes=n_pca_modes,
            pca_init_rank_reduction=pca_init_rank_reduction,
            alpha=alpha,
            compute=compute,
            sample_name=sample_name,
            feature_name=feature_name,
            solver=solver,
            random_state=random_state,
            solver_kwargs=solver_kwargs,
        )
        self.attrs.update({"model": "Complex Continuous Power CCA"})

        padding = self._process_parameter("padding", padding, "epx")
        decay_factor = self._process_parameter("decay_factor", decay_factor, 0.2)
        self._params["padding"] = padding
        self._params["decay_factor"] = decay_factor

    def _augment_data(self, X: DataArray, Y: DataArray) -> tuple[DataArray, DataArray]:
        """Augment the data with the Hilbert transform."""
        params = self.get_params()
        padding = params["padding"]
        decay_factor = params["decay_factor"]
        X = hilbert_transform(
            X,
            dims=(self.sample_name, self.feature_name[0]),
            padding=padding[0],
            decay_factor=decay_factor[0],
        )
        Y = hilbert_transform(
            Y,
            dims=(self.sample_name, self.feature_name[1]),
            padding=padding[1],
            decay_factor=decay_factor[1],
        )
        return X, Y

    def components_amplitude(self, normalized=True) -> Tuple[DataObject, DataObject]:
        """Compute the amplitude of the components.

        The amplitude of the components are defined as

        .. math::
            A_ij = |C_ij|

        where :math:`C_{ij}` is the :math:`i`-th entry of the :math:`j`-th component and
        :math:`|\\cdot|` denotes the absolute value.

        Returns
        -------
        DataObject
            Amplitude of the left components.
        DataObject
            Amplitude of the right components.

        """
        Px, Py = self._get_components(normalized=normalized)

        Px = self.whitener1.inverse_transform_components(Px)
        Py = self.whitener2.inverse_transform_components(Py)

        Px = abs(Px)
        Py = abs(Py)

        Px.name = "components_amplitude_X"
        Py.name = "components_amplitude_Y"

        Px = self.preprocessor1.inverse_transform_components(Px)
        Py = self.preprocessor2.inverse_transform_components(Py)

        return Px, Py

    def components_phase(self, normalized=True) -> Tuple[DataObject, DataObject]:
        """Compute the phase of the components.

        The phase of the components are defined as

        .. math::
            \\phi_{ij} = \\arg(C_{ij})

        where :math:`C_{ij}` is the :math:`i`-th entry of the :math:`j`-th component and
        :math:`\\arg(\\cdot)` denotes the argument of a complex number.

        Returns
        -------
        DataObject
            Phase of the left components.
        DataObject
            Phase of the right components.

        """
        Px, Py = self._get_components(normalized=normalized)

        Px = self.whitener1.inverse_transform_components(Px)
        Py = self.whitener2.inverse_transform_components(Py)

        Px = xr.apply_ufunc(np.angle, Px, keep_attrs=True)
        Py = xr.apply_ufunc(np.angle, Py, keep_attrs=True)

        Px.name = "components_phase_X"
        Py.name = "components_phase_Y"

        Px = self.preprocessor1.inverse_transform_components(Px)
        Py = self.preprocessor2.inverse_transform_components(Py)

        return Px, Py

    def scores_amplitude(self, normalized=False) -> Tuple[DataArray, DataArray]:
        """Compute the amplitude of the scores.

        The amplitude of the scores are defined as

        .. math::
            A_ij = |S_ij|

        where :math:`S_{ij}` is the :math:`i`-th entry of the :math:`j`-th score and
        :math:`|\\cdot|` denotes the absolute value.

        Returns
        -------
        DataArray
            Amplitude of the left scores.
        DataArray
            Amplitude of the right scores.

        """
        Rx, Ry = self._get_scores(normalized=normalized)

        Rx = self.whitener1.inverse_transform_scores(Rx)
        Ry = self.whitener2.inverse_transform_scores(Ry)

        Rx = abs(Rx)
        Ry = abs(Ry)

        Rx.name = "scores_amplitude_X"
        Ry.name = "scores_amplitude_Y"

        Rx = self.preprocessor1.inverse_transform_scores(Rx)
        Ry = self.preprocessor2.inverse_transform_scores(Ry)

        return Rx, Ry

    def scores_phase(self, normalized=False) -> Tuple[DataArray, DataArray]:
        """Compute the phase of the scores.

        The phase of the scores are defined as

        .. math::
            \\phi_{ij} = \\arg(S_{ij})

        where :math:`S_{ij}` is the :math:`i`-th entry of the :math:`j`-th score and
        :math:`\\arg(\\cdot)` denotes the argument of a complex number.

        Returns
        -------
        DataArray
            Phase of the left scores.
        DataArray
            Phase of the right scores.

        """
        Rx, Ry = self._get_scores(normalized=normalized)

        Rx = self.whitener1.inverse_transform_scores(Rx)
        Ry = self.whitener2.inverse_transform_scores(Ry)

        Rx = xr.apply_ufunc(np.angle, Rx, keep_attrs=True)
        Ry = xr.apply_ufunc(np.angle, Ry, keep_attrs=True)

        Rx.name = "scores_phase_X"
        Ry.name = "scores_phase_Y"

        Rx = self.preprocessor1.inverse_transform_scores(Rx)
        Ry = self.preprocessor2.inverse_transform_scores(Ry)

        return Rx, Ry

    def _transform_algorithm(self, data1: DataArray, data2: DataArray) -> dict:
        raise NotImplementedError(
            "The transform method is not implemented for Complex Continuous Power CCA."
        )


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
#         solver_kwargs: Dict = {},
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
