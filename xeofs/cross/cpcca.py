import warnings
from typing import Sequence

import numpy as np
import xarray as xr
from typing_extensions import Self

from ..linalg._numpy import _fractional_matrix_power
from ..linalg.decomposer import Decomposer
from ..utils.data_types import DataArray, DataObject
from ..utils.hilbert_transform import hilbert_transform
from ..utils.statistics import pearson_correlation
from ..utils.xarray_utils import argsort_dask
from .base_model_cross_set import BaseModelCrossSet


class CPCCA(BaseModelCrossSet):
    """Continuum Power CCA (CPCCA).

    CPCCA extends continuum power regression to isolate pairs of coupled
    patterns, maximizing the squared covariance between partially whitened
    variables [1]_ [2]_.

    This method solves the following optimization problem:

        :math:`\\max_{q_x, q_y} \\left( q_x^T X^T Y q_y \\right)`

    subject to the constraints:

        :math:`q_x^T (X^TX)^{1-\\alpha_x} q_x = 1, \\quad q_y^T
        (Y^TY)^{1-\\alpha_y} q_y = 1`

    where :math:`\\alpha_x` and :math:`\\alpha_y` control the degree of
    whitening applied to the data.

    Parameters
    ----------
    n_modes : int, default=2
        Number of modes to calculate.
    alpha : Sequence[float] | float, default=0.2
        Degree of whitening applied to the data. If float, the same value is
        applied to both data sets.
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

    Notes
    -----
    Canonical Correlation Analysis (CCA), Maximum Covariance Analysis (MCA) and
    Redundany Analysis (RDA) are all special cases of CPCCA depending on the
    choice of the parameter :math:`\\alpha`.

    References
    ----------
    .. [1] Swenson, E. Continuum Power CCA: A Unified Approach for Isolating
        Coupled Modes. Journal of Climate 28, 1016–1030 (2015).
    .. [2] Wilks, D. S. Statistical Methods in the Atmospheric Sciences.
        (Academic Press, 2019).
        doi:https://doi.org/10.1016/B978-0-12-815823-4.00011-0.

    Examples
    --------
    Perform regular CCA on two data sets:

    >>> model = CPCCA(n_modes=5, alpha=0.0)
    >>> model.fit(X, Y)

    Perform regularized CCA on two data sets:

    >>> model = CPCCA(n_modes=5, alpha=0.2)
    >>> model.fit(X, Y)

    Perform Maximum Covariance Analysis:

    >>> model = CPCCA(n_modes=5, alpha=1.0)
    >>> model.fit(X, Y)

    Perform Redundancy Analysis:

    >>> model = CPCCA(n_modes=5, alpha=[0, 1])
    >>> model.fit(X, Y)

    Make predictions for `Y` given `X`:

    >>> scores_y_pred = model.predict(X)  # prediction in "PC" space
    >>> Y_pred = model.inverse_transform(Y=scores_y_pred)  # prediction in physical space


    """

    def __init__(
        self,
        n_modes: int = 2,
        alpha: Sequence[float] | float = 0.2,
        standardize: Sequence[bool] | bool = False,
        use_coslat: Sequence[bool] | bool = False,
        use_pca: Sequence[bool] | bool = True,
        n_pca_modes: Sequence[float | int | str] | float | int | str = 0.999,
        pca_init_rank_reduction: Sequence[float] | float = 0.3,
        check_nans: Sequence[bool] | bool = True,
        compute: bool = True,
        sample_name: str = "sample",
        feature_name: Sequence[str] | str = "feature",
        solver: str = "auto",
        random_state: np.random.Generator | int | None = None,
        solver_kwargs: dict = {},
        **kwargs,
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
        self.attrs.update({"model": "Continuum Power CCA"})
        # Remove center from the inherited serialization params because it is hard-coded for CPCCA
        self._params.pop("center")

        params = self.get_params()
        self.sample_name: str = params["sample_name"]
        self.feature_name: tuple[str, str] = params["feature_name"]

    def _fit_algorithm(
        self,
        X: DataArray,
        Y: DataArray,
    ) -> Self:
        feature_name = self.feature_name

        # Compute the totalsquared covariance from the unwhitened data
        C_whitened = self._compute_cross_matrix(
            X,
            Y,
            sample_dim=self.sample_name,
            feature_dim_x=feature_name[0],
            feature_dim_y=feature_name[1],
            method="covariance",
            diagonal=False,
        )

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

        # Index of the sorted covariance explained
        idx_sorted_modes = argsort_dask(singular_values, "mode")[::-1]  # type: ignore
        idx_sorted_modes.coords.update(singular_values.coords)

        # Project the data onto the singular vectors
        scores1 = xr.dot(X, Q1, dims=feature_name[0])
        scores2 = xr.dot(Y, Q2, dims=feature_name[1])

        norm1 = np.sqrt(xr.dot(scores1.conj(), scores1, dims=self.sample_name)).real
        norm2 = np.sqrt(xr.dot(scores2.conj(), scores2, dims=self.sample_name)).real

        self.data.add(name="input_data1", data=X, allow_compute=False)
        self.data.add(name="input_data2", data=Y, allow_compute=False)
        self.data.add(name="components1", data=Q1)
        self.data.add(name="components2", data=Q2)
        self.data.add(name="scores1", data=scores1)
        self.data.add(name="scores2", data=scores2)
        self.data.add(name="singular_values", data=singular_values)
        self.data.add(name="squared_covariance", data=singular_values**2)
        self.data.add(name="total_squared_covariance", data=total_squared_covariance)
        self.data.add(name="idx_modes_sorted", data=idx_sorted_modes)
        self.data.add(name="norm1", data=norm1)
        self.data.add(name="norm2", data=norm2)

        # # Assign analysis-relevant meta data
        self.data.set_attrs(self.attrs)
        return self

    def _transform_algorithm(
        self,
        X: DataArray | None = None,
        Y: DataArray | None = None,
        normalized=False,
    ) -> dict[str, DataArray]:
        results = {}
        if X is not None:
            # Project data onto singular vectors
            comps1 = self.data["components1"]
            norm1 = self.data["norm1"]
            scores1 = xr.dot(X, comps1)
            if normalized:
                scores1 = scores1 / norm1
            results["X"] = scores1

        if Y is not None:
            # Project data onto singular vectors
            comps2 = self.data["components2"]
            norm2 = self.data["norm2"]
            scores2 = xr.dot(Y, comps2)
            if normalized:
                scores2 = scores2 / norm2
            results["Y"] = scores2

        return results

    def _inverse_transform_algorithm(
        self, X: DataArray | None = None, Y: DataArray | None = None
    ) -> dict[str, DataArray]:
        x_is_given = X is not None
        y_is_given = Y is not None

        if (not x_is_given) and (not y_is_given):
            raise ValueError("Either X or Y must be given.")

        results = {}

        if x_is_given:
            # Singular vectors
            comps1 = self.data["components1"].sel(mode=X.mode)
            # Reconstruct the data
            results["X"] = xr.dot(X, comps1.conj(), dims="mode")

        if y_is_given:
            # Singular vectors
            comps2 = self.data["components2"].sel(mode=Y.mode)
            # Reconstruct the data
            results["Y"] = xr.dot(Y, comps2.conj(), dims="mode")

        return results

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

    def _get_components(self, normalized=True):
        comps1 = self.data["components1"]
        comps2 = self.data["components2"]

        if not normalized:
            comps1 = comps1 * self.data["norm1"]
            comps2 = comps2 * self.data["norm2"]

        return comps1, comps2

    def _get_scores(self, normalized=False):
        norm1 = self.data["norm1"]
        norm2 = self.data["norm2"]

        scores1 = self.data["scores1"]
        scores2 = self.data["scores2"]

        if normalized:
            scores1 = scores1 / norm1
            scores2 = scores2 / norm2

        return scores1, scores2

    def cross_correlation_coefficients(self):
        """Get the cross-correlation coefficients.

        The cross-correlation coefficients between the scores of ``X`` and ``Y``
        are computed as:

        .. math::
            c_{xy, i} = \\text{corr} \\left(\\mathbf{r}_{x, i}, \\mathbf{r}_{y, i} \\right)

        where :math:`\\mathbf{r}_{x, i}` and :math:`\\mathbf{r}_{y, i}` are the
        `i`th scores of ``X`` and ``Y``,

        Notes
        -----
        When :math:`\\alpha=0`, the cross-correlation coefficients are
        equivalent to the canonical correlation coefficients.

        """

        Rx = self.data["scores1"]
        Ry = self.data["scores2"]

        cross_corr = self._compute_cross_matrix(
            Rx,
            Ry,
            sample_dim=self.sample_name,
            feature_dim_x="mode",
            feature_dim_y="mode",
            method="correlation",
            diagonal=True,
        )
        cross_corr = cross_corr.real
        cross_corr.name = "cross_correlation_coefficients"
        return cross_corr

    def correlation_coefficients_X(self):
        """Get the correlation coefficients for the scores of :math:`X`.

        The correlation coefficients of the scores of :math:`X` are given by:

        .. math::
            c_{x, ij} = \\text{corr} \\left(\\mathbf{r}_{x, i}, \\mathbf{r}_{x, j} \\right)

        where :math:`\\mathbf{r}_{x, i}` and :math:`\\mathbf{r}_{x, j}` are the
        `i`th and `j`th scores of :math:`X`.

        """
        Rx = self.data["scores1"]

        corr = self._compute_cross_matrix(
            Rx,
            Rx,
            sample_dim=self.sample_name,
            feature_dim_x="mode",
            feature_dim_y="mode",
            method="correlation",
            diagonal=False,
        )
        corr.name = "correlation_coefficients_X"
        return corr

    def correlation_coefficients_Y(self):
        """Get the correlation coefficients for the scores of :math:`Y`.

        The correlation coefficients of the scores of :math:`Y` are given by:

        .. math::
            c_{y, ij} = \\text{corr} \\left(\\mathbf{r}_{y, i}, \\mathbf{r}_{y, j} \\right)

        where :math:`\\mathbf{r}_{y, i}` and :math:`\\mathbf{r}_{y, j}` are the
        `i`th and `j`th scores of :math:`Y`.

        """
        Ry = self.data["scores2"]

        corr = self._compute_cross_matrix(
            Ry,
            Ry,
            sample_dim=self.sample_name,
            feature_dim_x="mode",
            feature_dim_y="mode",
            method="correlation",
            diagonal=False,
        )
        corr.name = "correlation_coefficients_Y"
        return corr

    def squared_covariance_fraction(self):
        """Get the squared covariance fraction (SCF).

        The SCF is computed as a weighted mean-square error (see equation (15)
        in Swenson (2015)) :

        .. math::
            SCF_{i} = 1 - \\frac{\\|\\mathbf{d}_{X,i}^T \\mathbf{d}_{Y,i}\\|_F^2}{\\|X^TY\\|_F^2}

        where :math:`\\mathbf{d}_{X,i}` and :math:`\\mathbf{d}_{Y,i}` are the
        residuals of the input data :math:`X` and :math:`Y` after reconstruction
        by the `ith` scores of :math:`X` and :math:`Y`, respectively.

        References
        ----------
        Swenson, E. Continuum Power CCA: A Unified Approach for Isolating
            Coupled Modes. Journal of Climate 28, 1016–1030 (2015).

        """

        def _compute_residual_variance_numpy(X, Y, Xrec, Yrec):
            dX = X - Xrec
            dY = Y - Yrec

            return np.linalg.norm(dX.conj().T @ dY / (dX.shape[0] - 1)) ** 2

        total_squared_covariance = self.data["total_squared_covariance"]
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

        # Rename the sample dimension to avoid conflicts for
        # different coordinates with same length
        X1 = X1.rename({self.sample_name: sample_name_x})
        X2 = X2.rename({self.sample_name: sample_name_y})

        # Get the component scores
        scores1 = self.data["scores1"]
        scores2 = self.data["scores2"]

        # Compute the residual variance for each mode
        squared_covariance_fraction: list[DataArray] = []
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
            squared_covariance_fraction.append(1 - res_var / total_squared_covariance)

        scf = xr.concat(squared_covariance_fraction, dim="mode")
        scf.name = "squared_covariance_fraction"

        # In theory, the residual can be larger than the total squared covariance
        # if a mode is not well defined. In this case, the SCF would be negative.
        # We set these values to zero.
        scf = xr.where(scf < 0, 0, scf)
        return scf

    def fraction_variance_X_explained_by_X(self):
        """Get the fraction of variance explained (FVE X).

        The FVE X is the fraction of variance in :math:`X` explained by the
        scores of :math:`X`. It is computed as a weighted mean-square error (see
        equation (15) in Swenson (2015)) :

        .. math::
            FVE_{X|X,i} = 1 - \\frac{\\|\\mathbf{d}_{X,i}\\|_F^2}{\\|X\\|_F^2}

        where :math:`\\mathbf{d}_{X,i}` are the residuals of the input data
        :math:`X` after reconstruction by the `ith` scores of :math:`X`.

        References
        ----------
        Swenson, E. Continuum Power CCA: A Unified Approach for Isolating
            Coupled Modes. Journal of Climate 28, 1016–1030 (2015).

        """
        # Get the singular vectors
        Qx = self.data["components1"]

        # Get input data
        X = self.data["input_data1"]

        # Unwhiten the data
        X = self.whitener1.inverse_transform_data(X, unwhiten_only=True)

        # Compute the total variance
        total_variance: DataArray = self._compute_total_variance(X, self.sample_name)

        # Get the component scores
        Rx = self.data["scores1"]

        # Compute the residual variance for each mode
        fraction_variance_explained: list[DataArray] = []
        for mode in Rx.mode.values:
            # Reconstruct the data
            Xr = xr.dot(Rx.sel(mode=[mode]), Qx.sel(mode=[mode]).conj().T, dims="mode")

            # Unwhitend the reconstructed data
            Xr = self.whitener1.inverse_transform_data(Xr, unwhiten_only=True)

            # Compute fraction variance explained
            residual_variance = self._compute_total_variance(X - Xr, self.sample_name)
            residual_variance = residual_variance.expand_dims({"mode": [mode]})
            fraction_variance_explained.append(1 - residual_variance / total_variance)

        fve_xx = xr.concat(fraction_variance_explained, dim="mode")
        fve_xx.name = "fraction_variance_X_explained_by_X"
        return fve_xx

    def fraction_variance_Y_explained_by_Y(self):
        """Get the fraction of variance explained (FVE Y).

        The FVE Y is the fraction of variance in :math:`Y` explained by the
        scores of :math:`Y`. It is computed as a weighted mean-square error (see
        equation (15) in Swenson (2015)) :

        .. math::
            FVE_{Y|Y,i} = 1 - \\frac{\\|\\mathbf{d}_{Y,i}\\|_F^2}{\\|Y\\|_F^2}

        where :math:`\\mathbf{d}_{Y,i}` are the residuals of the input data
        :math:`Y` after reconstruction by the `ith` scores of :math:`Y`.

        References
        ----------
        Swenson, E. Continuum Power CCA: A Unified Approach for Isolating
            Coupled Modes. Journal of Climate 28, 1016–1030 (2015).

        """
        # Get the singular vectors
        Qy = self.data["components2"]

        # Get input data
        Y = self.data["input_data2"]

        # Unwhiten the data
        Y = self.whitener2.inverse_transform_data(Y, unwhiten_only=True)

        # Compute the total variance
        total_variance: DataArray = self._compute_total_variance(Y, self.sample_name)

        # Get the component scores
        Ry = self.data["scores2"]

        # Compute the residual variance for each mode
        fraction_variance_explained: list[DataArray] = []
        for mode in Ry.mode.values:
            # Reconstruct the data
            Yr = xr.dot(Ry.sel(mode=[mode]), Qy.sel(mode=[mode]).conj().T, dims="mode")

            # Unwhitend the reconstructed data
            Yr = self.whitener2.inverse_transform_data(Yr, unwhiten_only=True)

            # Compute fraction variance explained
            residual_variance = self._compute_total_variance(Y - Yr, self.sample_name)
            residual_variance = residual_variance.expand_dims({"mode": [mode]})
            fraction_variance_explained.append(1 - residual_variance / total_variance)

        fve_yy = xr.concat(fraction_variance_explained, dim="mode")
        fve_yy.name = "fraction_variance_Y_explained_by_Y"
        return fve_yy

    def fraction_variance_Y_explained_by_X(self) -> DataArray:
        """Get the fraction of variance explained (FVE YX).

        The FVE YX is the fraction of variance in :math:`Y` explained by the
        scores of :math:`X`. It is computed as a weighted mean-square error (see
        equation (15) in Swenson (2015)) :

        .. math::
            FVE_{Y|X,i} = 1 - \\frac{\\|(X^TX)^{-1/2} \\mathbf{d}_{X,i}^T \\mathbf{d}_{Y,i}\\|_F^2}{\\|(X^TX)^{-1/2} X^TY\\|_F^2}

        where :math:`\\mathbf{d}_{X,i}` and :math:`\\mathbf{d}_{Y,i}` are the
        residuals of the input data :math:`X` and :math:`Y` after reconstruction
        by the `ith` scores of :math:`X` and :math:`Y`, respectively.

        References
        ----------
        Swenson, E. Continuum Power CCA: A Unified Approach for Isolating
        Coupled Modes. Journal of Climate 28, 1016–1030 (2015).

        """

        def _compute_total_variance_numpy(X, Y):
            Cx = X.conj().T @ X / (X.shape[0] - 1)
            Tinv = _fractional_matrix_power(Cx, -0.5)
            return np.linalg.norm(Tinv @ X.conj().T @ Y / (X.shape[0] - 1)) ** 2

        def _compute_residual_variance_numpy(X, Y, Xrec, Yrec):
            dX = X - Xrec
            dY = Y - Yrec

            Cx = X.conj().T @ X / (X.shape[0] - 1)
            Tinv = _fractional_matrix_power(Cx, -0.5)
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
        """Get the homogeneous correlation patterns.

        The homogeneous correlation patterns are the correlation coefficients
        between the input data and the scores. They are defined as:

        .. math::
            H_{X, i} = \\text{corr} \\left(X, \\mathbf{r}_{x,i} \\right)

        .. math::
            H_{Y, i} = \\text{corr} \\left(Y, \\mathbf{r}_{y,i} \\right)

        where :math:`X` and :math:`Y` are the input data, and
        :math:`\\mathbf{r}_{x,i}` and :math:`\\mathbf{r}_{y,i}` are the `i`th
        scores of :math:`X` and :math:`Y`, respectively.


        Parameters
        ----------
        correction: str, default=None
            Method to apply a multiple testing correction. If None, no
            correction is applied.  Available methods are: - bonferroni :
            one-step correction - sidak : one-step correction - holm-sidak :
            step down method using Sidak adjustments - holm : step-down method
            using Bonferroni adjustments - simes-hochberg : step-up method
            (independent) - hommel : closed method based on Simes tests
            (non-negative) - fdr_bh : Benjamini/Hochberg (non-negative)
            (default) - fdr_by : Benjamini/Yekutieli (negative) - fdr_tsbh : two
            stage fdr correction (non-negative) - fdr_tsbky : two stage fdr
            correction (non-negative)
        alpha: float, default=0.05
            The desired family-wise error rate. Not used if `correction` is
            None.

        Returns
        -------
        tuple[DataObject, DataObject]
            Homogenous correlation patterns of `X` and `Y`.
        tuple[DataObject, DataObject]
            p-values of the homogenous correlation patterns of `X` and `Y`.

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
        """Get the heterogeneous correlation patterns.

        The heterogeneous patterns are the correlation coefficients between the
        input data and the scores of the other field:

        .. math::
            G_{X, i} = \\text{corr} \\left(X, \\mathbf{r}_{y,i} \\right)

        .. math::
            G_{Y, i} = \\text{corr} \\left(Y, \\mathbf{r}_{x,i} \\right)

        where :math:`X` and :math:`Y` are the input data, and
        :math:`\\mathbf{r}_{x,i}` and :math:`\\mathbf{r}_{y,i}` are the `i`th
        scores of :math:`X` and :math:`Y`, respectively.

        Parameters
        ----------
        correction: str, default=None
            Method to apply a multiple testing correction. If None, no
            correction is applied.  Available methods are: - bonferroni :
            one-step correction - sidak : one-step correction - holm-sidak :
            step down method using Sidak adjustments - holm : step-down method
            using Bonferroni adjustments - simes-hochberg : step-up method
            (independent) - hommel : closed method based on Simes tests
            (non-negative) - fdr_bh : Benjamini/Hochberg (non-negative)
            (default) - fdr_by : Benjamini/Yekutieli (negative) - fdr_tsbh : two
            stage fdr correction (non-negative) - fdr_tsbky : two stage fdr
            correction (non-negative)
        alpha: float, default=0.05
            The desired family-wise error rate. Not used if `correction` is
            None.

        Returns
        -------
        tuple[DataObject, DataObject]
            Heterogenous correlation patterns of `X` and `Y`.
        tuple[DataObject, DataObject]
            p-values of the heterogenous correlation patterns of `X` and `Y`.

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
            feature_name=self.feature_name[1],
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

    def _compute_cross_matrix(
        self,
        X: DataArray,
        Y: DataArray,
        sample_dim: str,
        feature_dim_x: str,
        feature_dim_y: str,
        method: str = "covariance",
        diagonal: bool = False,
    ) -> DataArray:
        """Compute the cross matrix of two data objects.

        Assume centered data.

        Parameters
        ----------
        X, Y : DataArray
            DataArrays to compute the cross matrix from.
        sample_dim : str
            Name of the sample dimension.
        feature_dim_x, feature_dim_y : str
            Name of the feature dimensions. If the feature dimensions are the same, they are renamed to avoid conflicts.
        method : {"covariance", "correlation"}
            Method to compute the cross matrix.
        diagonal : bool, default=False
            Whether to compute the diagonal of the cross matrix.

        Returns
        -------
        DataArray
            The cross matrix of the two data objects.

        """
        if feature_dim_x == feature_dim_y:
            new_feature_dim_x = feature_dim_x + "_x"
            new_feature_dim_y = feature_dim_y + "_y"
            X = X.rename({feature_dim_x: new_feature_dim_x})
            Y = Y.rename({feature_dim_y: new_feature_dim_y})
            feature_dim_x = new_feature_dim_x
            feature_dim_y = new_feature_dim_y

        # Rename the sample dimension to avoid conflicts for
        # different coordinates with same length
        sample_dim_x = sample_dim + "_x"
        sample_dim_y = sample_dim + "_y"
        X = X.rename({sample_dim: sample_dim_x})
        Y = Y.rename({sample_dim: sample_dim_y})

        if method == "correlation":
            X = self._normalize_data(X, sample_dim_x)
            Y = self._normalize_data(Y, sample_dim_y)

        if diagonal:
            return xr.apply_ufunc(
                self._compute_cross_covariance_diagonal_numpy,
                X,
                Y,
                input_core_dims=[
                    [sample_dim_x, feature_dim_x],
                    [sample_dim_y, feature_dim_y],
                ],
                output_core_dims=[[feature_dim_y]],
                dask="allowed",
            ).rename({feature_dim_y: feature_dim_y[:-2]})
        else:
            return xr.apply_ufunc(
                self._compute_cross_covariance_numpy,
                X,
                Y,
                input_core_dims=[
                    [sample_dim_x, feature_dim_x],
                    [sample_dim_y, feature_dim_y],
                ],
                output_core_dims=[[feature_dim_x, feature_dim_y]],
                dask="allowed",
            )

    def _compute_cross_covariance_diagonal_numpy(self, X, Y):
        # Assume centered data
        return np.diag(self._compute_cross_covariance_numpy(X, Y))

    def _compute_total_squared_covariance(self, C: DataArray) -> DataArray:
        """Compute the total squared covariance.

        Requires the unwhitened covariance matrix which we can obtain by multiplying the whitened covariance matrix with the inverse of the whitening transformation matrix.
        """
        C = self.whitener2.inverse_transform_data(C)
        C = self.whitener1.inverse_transform_data(C.conj().T)
        # Not necessary to conjugate transpose for total squared covariance
        # C = C.conj().T
        return (abs(C) ** 2).sum()

    @staticmethod
    def _compute_total_variance(X: DataArray, dim: str) -> DataArray:
        """Compute the total variance of the centered data."""
        return (X * X.conj()).sum() / (X[dim].size - 1)

    @staticmethod
    def _compute_cross_covariance_numpy(X, Y):
        # Assume centered data
        n_samples_x = X.shape[0]
        n_samples_y = Y.shape[0]
        if n_samples_x != n_samples_y:
            err_msg = f"Both data matrices must have the same number of samples but found {n_samples_x} in the first and {n_samples_y} in the second."
            raise ValueError(err_msg)
        return X.conj().T @ Y / (n_samples_x - 1)

    @staticmethod
    def _normalize_data(X, dim):
        # Assume centered data
        return X / X.std(dim)


class ComplexCPCCA(CPCCA):
    """Complex CPCCA.

    Complex CPCCA extends classical CPCCA [1]_ by examining amplitude-phase
    relationships. It is based on complex-valued fields obtained from a pair of
    variables such as the zonal and meridional components, :math:`U` and
    :math:`V`, of the wind field, leading to complex-valued data matrices:

        .. math::
        X = U_x + iV_x

    and

    .. math::
        Y = U_y + iV_y

    This method solves the following optimization problem:

        :math:`\\max_{q_x, q_y} \\left( q_x^H X^H Y q_y \\right)`

    subject to the constraints:

        :math:`q_x^H (X^HX)^{1-\\alpha_x} q_x = 1, \\quad q_y^H
        (Y^HY)^{1-\\alpha_y} q_y = 1`

    where :math:`H` denotes the conjugate transpose, :math:`X` and :math:`Y` are
    the complex-valued data matrices, and :math:`\\alpha_x` and
    :math:`\\alpha_y` control the degree of whitening applied to the data.

    Parameters
    ----------
    n_modes : int, default=2
        Number of modes to calculate.
    alpha : Sequence[float] | float, default=0.2
        Degree of whitening applied to the data. If float, the same value is
        applied to both data sets.
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

    With two DataArrays :math:`u_i` and :math:`v_i` representing the zonal and
    meridional components of the wind field for two different regions :math:`x`
    and :math:`y`, construct

    >>> X = u_x + 1j * v_x
    >>> Y = u_y + 1j * v_y

    and fit the Complex CPCCA model:

    >>> model = ComplexCPCCA(n_modes=5)
    >>> model.fit(X, Y, "time")

    Finally, extract the amplitude and phase patterns:

    >>> amp_x, amp_y = model.components_amplitude()
    >>> phase_x, phase_y = model.components_phase()


    References
    ----------
    .. [1] Swenson, E. Continuum Power CCA: A Unified Approach for Isolating
        Coupled Modes. Journal of Climate 28, 1016–1030 (2015).

    """

    def __init__(
        self,
        n_modes: int = 2,
        alpha: Sequence[float] | float = 0.2,
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
        self.attrs.update({"model": "Complex CPCCA"})

    def _fit_algorithm(self, X: DataArray, Y: DataArray) -> Self:
        if (not np.iscomplexobj(X)) or (not np.iscomplexobj(Y)):
            warnings.warn(
                "Expected complex-valued data but found real-valued data. For Hilbert model, use corresponding `Hilbert` class."
            )

        return super()._fit_algorithm(X, Y)

    def components_amplitude(self, normalized=True) -> tuple[DataObject, DataObject]:
        """Get the amplitude of the components.

        The amplitudes of the components are defined as

        .. math::
            A_{x, ij} = |p_{x, ij}|
        .. math::
            A_{y, ij} = |p_{y, ij}|

        where :math:`p_{ij}` is the :math:`i`-th entry of the :math:`j`-th
        component and :math:`|\\cdot|` denotes the absolute value.

        Returns
        -------
        tuple[DataObject, DataObject]
            Component amplitudes of :math:`X` and :math:`Y`.

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

    def components_phase(self, normalized=True) -> tuple[DataObject, DataObject]:
        """Get the phase of the components.

        The phases of the components are defined as

        .. math::
            \\phi_{x, ij} = \\arg(p_{x, ij})
        .. math::
            \\phi_{y, ij} = \\arg(p_{y, ij})

        where :math:`p_{ij}` is the :math:`i`-th entry of the :math:`j`-th component and
        :math:`\\arg(\\cdot)` denotes the argument of a complex number.

        Returns
        -------
        tuple[DataObject, DataObject]
            Component phases of :math:`X` and :math:`Y`.

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

    def scores_amplitude(self, normalized=False) -> tuple[DataArray, DataArray]:
        """Get the amplitude of the scores.

        The amplitudes of the scores are defined as

        .. math::
            A_{x, ij} = |r_{y, ij}|
        .. math::
            A_{y, ij} = |r_{x, ij}|

        where :math:`r_{ij}` is the :math:`i`-th entry of the :math:`j`-th score and
        :math:`|\\cdot|` denotes the absolute value.

        Returns
        -------
        tuple[DataArray, DataArray]
            Score amplitudes of :math:`X` and :math:`Y`.

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

    def scores_phase(self, normalized=False) -> tuple[DataArray, DataArray]:
        """Get the phase of the scores.

        The phases of the scores are defined as

        .. math::
            \\phi_{x, ij} = \\arg(r_{x, ij})
        .. math::
            \\phi_{y, ij} = \\arg(r_{y, ij})

        where :math:`r_{ij}` is the :math:`i`-th entry of the :math:`j`-th score and
        :math:`\\arg(\\cdot)` denotes the argument of a complex number.

        Returns
        -------
        tuple[DataArray, DataArray]
            Score phases of :math:`X` and :math:`Y`.

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


class HilbertCPCCA(ComplexCPCCA):
    """Hilbert CPCCA.

    Hilbert CPCCA extends classical CPCCA [1]_ by examining
    amplitude-phase relationships. It augments the input data with its Hilbert
    transform, creating a complex-valued field.

    This method solves the following optimization problem:

        :math:`\\max_{q_x, q_y} \\left( q_x^H X^H Y q_y \\right)`

    subject to the constraints:

        :math:`q_x^H (X^HX)^{1-\\alpha_x} q_x = 1, \\quad q_y^H
        (Y^HY)^{1-\\alpha_y} q_y = 1`

    where :math:`H` denotes the conjugate transpose, :math:`X` and :math:`Y` are
    the augmented data matrices, and :math:`\\alpha_x` and :math:`\\alpha_y`
    control the degree of whitening applied to the data.

    Parameters
    ----------
    n_modes : int, default=2
        Number of modes to calculate.
    alpha : Sequence[float] | float, default=0.2
        Degree of whitening applied to the data. If float, the same value is
        applied to both data sets.
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

    Perform Hilbert CPCCA on two real-valued datasets `X` and `Y`, using
    exponential padding:

    >>> model = HilbertCPCCA(n_modes=5, padding="exp")
    >>> model.fit(X, Y)

    References
    ----------
    .. [1] Swenson, E. Continuum Power CCA: A Unified Approach for Isolating
        Coupled Modes. Journal of Climate 28, 1016–1030 (2015).

    """

    def __init__(
        self,
        n_modes: int = 2,
        alpha: Sequence[float] | float = 0.2,
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
        ComplexCPCCA.__init__(
            self,
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
        self.attrs.update({"model": "Hilbert CPCCA"})

        padding = self._process_parameter("padding", padding, "epx")
        decay_factor = self._process_parameter("decay_factor", decay_factor, 0.2)
        self._params["padding"] = padding
        self._params["decay_factor"] = decay_factor

    def _fit_algorithm(self, X: DataArray, Y: DataArray) -> Self:
        CPCCA._fit_algorithm(self, X, Y)
        return self

    def transform(
        self, X: DataObject | None = None, Y: DataObject | None = None, normalized=False
    ) -> Sequence[DataArray]:
        """Transform the input data into the component space."""
        raise NotImplementedError("Hilbert models do not support the transform method.")

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
