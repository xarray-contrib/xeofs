"""
This code is based on the work of James Chapman from cca-zoo.
Source: https://github.com/jameschapman19/cca_zoo

The original code is licensed under the MIT License.

Copyright (c) 2020 onward James Chapman
"""

from abc import abstractmethod
from datetime import datetime
from typing import Hashable, List, Sequence

import dask.array as da
import numpy as np
import xarray as xr
from scipy.linalg import eigh
from sklearn.base import BaseEstimator
from sklearn.utils.validation import FLOAT_DTYPES
from typing_extensions import Self

from xeofs.models import EOF

from .._version import __version__
from ..preprocessing.preprocessor import Preprocessor
from ..utils.data_types import DataArray, DataList, DataObject


def _check_parameter_number(parameter_name: str, parameter, n_views: int):
    if len(parameter) != n_views:
        raise ValueError(
            f"number of views passed should match number of parameter {parameter_name}"
            f"len(views)={n_views} and "
            f"len({parameter_name})={len(parameter)}"
        )


def _process_parameter(parameter_name: str, parameter, default, n_views: int):
    if parameter is None:
        parameter = [default] * n_views
    elif not isinstance(parameter, (list, tuple)):
        parameter = [parameter] * n_views
    _check_parameter_number(parameter_name, parameter, n_views)
    return parameter


class CCABaseModel(BaseEstimator):
    def __init__(
        self,
        n_modes: int = 10,
        use_coslat: bool = False,
        check_nans: bool = True,
        pca: bool = False,
        variance_fraction: float = 0.99,
        init_pca_modes: int | float = 0.75,
        compute: bool = True,
        sample_name: str = "sample",
        feature_name: str = "feature",
    ):
        self.sample_name = sample_name
        self.feature_name = feature_name
        self.n_modes = n_modes
        self.use_coslat = use_coslat
        self.pca = pca
        self.compute = compute
        self.variance_fraction = variance_fraction
        self.init_pca_modes = init_pca_modes

        self.dtypes = FLOAT_DTYPES

        self._preprocessor_kwargs = {
            "sample_name": sample_name,
            "feature_name": feature_name,
            "with_std": False,
            "check_nans": check_nans,
        }

        # Define analysis-relevant meta data
        self.attrs = {"model": "BaseCrossModel"}
        self.attrs.update(
            {
                "software": "xeofs",
                "version": __version__,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        # Initialize the data container only to avoid type errors
        # The actual data container will be initialized in respective subclasses
        # self.data: _BaseCrossModelDataContainer = _BaseCrossModelDataContainer()
        self.data = {}

    def _validate_data(self, views: Sequence[DataArray]):
        if not all(
            data[self.sample_name].size == views[0][self.sample_name].size
            for data in views
        ):
            raise ValueError("All views must have the same number of samples")
        if not all(data.ndim == 2 for data in views):
            raise ValueError("All views must have 2 dimensions")
        if not all(data.dtype in self.dtypes for data in views):
            raise ValueError("All views must have dtype of {}.".format(self.dtypes))
        if not all(data[self.feature_name].size >= self.n_modes for data in views):
            raise ValueError(
                "All views must have at least {} features.".format(self.n_modes)
            )

    def _process_init_pca_modes(self, n_modes):
        err_msg = "init_pca_modes must be either a float <= 1.0 or an integer > 1"
        n_modes_list = []
        n_modes_max = [
            min(self.n_samples_, n_features) for n_features in self.n_features_
        ]
        for n, n_max in zip(n_modes, n_modes_max):
            if isinstance(n, float):
                if n > 1.0:
                    raise ValueError(err_msg)
                n = int(n * n_max)
                n_modes_list.append(n)
            elif isinstance(n, int):
                if n <= 1:
                    raise ValueError(err_msg)
                n_modes_list.append(n)
            else:
                raise ValueError(err_msg)
        return n_modes_list

    def fit(
        self,
        views: Sequence[DataObject],
        dim: Hashable | Sequence[Hashable],
    ) -> Self:
        self.n_views_ = len(views)
        self.use_coslat = _process_parameter(
            "use_coslat", self.use_coslat, False, self.n_views_
        )
        self.init_pca_modes = _process_parameter(
            "init_pca_modes", self.init_pca_modes, 0.75, self.n_views_
        )

        # Preprocess the input data
        self.preprocessors = [
            Preprocessor(with_coslat=self.use_coslat[i], **self._preprocessor_kwargs)
            for i in range(self.n_views_)
        ]
        views2D: List[DataArray] = [
            preprocessor.fit_transform(data, dim)
            for preprocessor, data in zip(self.preprocessors, views)
        ]
        self._validate_data(views2D)
        self.n_features_ = [data.coords[self.feature_name].size for data in views2D]
        self.n_samples_ = views2D[0][self.sample_name].size

        self.data["input_data"] = views2D
        views2D = self._process_data(views2D)
        self.data["pca_data"] = views2D

        self._fit_algorithm(views2D)

        return self

    def _process_data(self, views: DataList) -> DataList:
        if self.pca:
            views = self._apply_pca(views)
        return views

    def _apply_pca(self, views: DataList):
        self.pca_models = []

        n_pca_modes = self._process_init_pca_modes(self.init_pca_modes)

        view_transformed = []

        for i, view in enumerate(views):
            # NOTE: coslat weighting already happens in Preprocessor class
            pca = EOF(n_modes=n_pca_modes[i], compute=self.compute)
            pca.fit(view, dim=self.sample_name)
            if self.compute:
                pca.compute()
            self.pca_models.append(pca)

            # TODO: method to get cumulative explained variance
            cum_exp_var_ratio = pca.explained_variance_ratio().cumsum()
            # Ensure that the sum of the explained variance ratio is always less than 1
            # Due to rounding errors the total sum may be slightly larger than 1,
            # which we counter by a small correction
            cum_exp_var_ratio -= 1e-6
            max_exp_var_ratio = cum_exp_var_ratio.isel(mode=-1).item()
            if (
                max_exp_var_ratio <= self.variance_fraction
                and max_exp_var_ratio <= 0.9999
            ):
                print(
                    "Warning: variance fraction {:.4f} is not reached. ".format(
                        self.variance_fraction
                    )
                    + "Only {:.4f} of variance is explained.".format(
                        cum_exp_var_ratio.isel(mode=-1).item()
                    )
                )
            n_modes_keep = (
                cum_exp_var_ratio.where(
                    cum_exp_var_ratio <= self.variance_fraction, drop=True
                ).size
                + 1
            )
            # Take at least 2 modes
            n_modes_keep = max(n_modes_keep, 2)

            scores = pca.scores(normalized=False).isel(mode=slice(0, n_modes_keep))
            scores = scores.rename({"mode": self.feature_name}).transpose(
                self.sample_name, self.feature_name
            )
            view_transformed.append(scores)
        return view_transformed

    @abstractmethod
    def _fit_algorithm(self, views: List[DataArray]) -> Self:
        raise NotImplementedError


class CCA(CCABaseModel):
    r"""Canonical Correlation Analysis.
    
    Canonical Correlation Analysis (CCA) identifies linear combinations of variables from multiple datasets that 
    maximize their mutual correlations. An optional regularisation parameter (ridge regression)
    can be used to improve the conditioning of the covariance matrix.

    The objective function of (regularised) CCA is:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}\\

        \text{subject to:}

        (1-c_1)w_1^TX_1^TX_1w_1+c_1w_1^Tw_1=n

        (1-c_2)w_2^TX_2^TX_2w_2+c_2w_2^Tw_2=n

    where :math:`c_i` are the regularization parameters for dataset.

    Parameters
    ----------
    n_modes : int, optional
        Number of latent dimensions to use, by default 10
    use_coslat : bool, optional
        Whether to use the square root of the cosine of the latitude as weights, by default False
    pca : bool, optional
        Whether to perform PCA on the input data, by default True
    variance_fraction : float, optional
        Fraction of variance to keep when performing PCA, by default 0.99
    init_pca_modes : int | float, optional
        Number of PCA modes to compute. If float, the number of modes is given by the fraction of maximum number of modes for the given data.
        A value of 1.0 will perform a full SVD of the data. Choosing a smaller value can increase computation speed. Default 0.75
    c : Sequence[float] | float], optional
        Regularisation parameter, by default 0 (no regularization)
    compute : bool, optional
        Whether to compute the decomposition immediately, by default True


    Notes
    -----
    This implementation is largely based on the MCCA class from the cca_zoo repository [3]_ .
    

    References
    ----------
    .. [1] Vinod, Hrishikesh _D. "Canonical ridge and econometrics of joint production." Journal of econometrics 4.2 (1976): 147-166.
    .. [2] Hotelling, Harold. "Relations between two sets of variates." Breakthroughs in statistics. Springer, New York, NY, 1992. 162-190.
    .. [3] Chapman et al., (2021). CCA-Zoo: A collection of Regularized, Deep Learning based, Kernel, and Probabilistic CCA methods in a scikit-learn style framework. Journal of Open Source Software, 6(68), 3823

    Examples
    --------
    >>> from xe.models import CCA
    >>> model = CCA(n_modes=5)
    >>> model.fit(data)
    >>> can_loadings = model.canonical_loadings()

    """

    def __init__(
        self,
        n_modes: int = 2,
        use_coslat: bool = False,
        check_nans: bool = True,
        c: float = 0,
        pca: bool = True,
        variance_fraction: float = 0.99,
        init_pca_modes: float = 0.75,
        compute: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__(
            n_modes=n_modes,
            use_coslat=use_coslat,
            check_nans=check_nans,
            pca=pca,
            compute=compute,
            variance_fraction=variance_fraction,
            init_pca_modes=init_pca_modes,
        )
        self.attrs.update({"model": "CCA"})
        self.c = c
        self.eps = eps

    def _fit_algorithm(self, views: List[DataArray]) -> Self:
        self.c = _process_parameter("c", self.c, 0, self.n_views_)
        eigvals, eigvecs = self._solve_gevp(views)
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        # Compute the weights for each view
        self._weights(eigvals, eigvecs, views)
        # Compute loadings (= normalized weights)
        self.data["loadings"] = [
            wght / self._apply_norm(wght, [self.feature_name])
            for wght in self.data["weights"]
        ]
        canonical_variates = self._transform(self.data["input_data"])
        self.data["variates"] = canonical_variates

        self.data["canonical_loadings"] = [
            xr.dot(data, vari, dims=self.sample_name, optimize=True)
            for data, vari in zip(self.data["input_data"], canonical_variates)
        ]

        # Compute explained variance
        # Transform the views using the loadings
        transformed_views = [
            xr.dot(view, loading, dims=self.feature_name)
            for view, loading in zip(self.data["input_data"], self.data["loadings"])
        ]
        # Calculate the variance of each latent dimension in the transformed views
        self.data["explained_variance"] = [
            transformed.var(self.sample_name) for transformed in transformed_views
        ]

        # Explained variance ratio
        self.data["total_variance"] = [
            view.var(self.sample_name, ddof=1).sum() for view in views
        ]

        # Calculate the explained variance ratio for each latent dimension for each view
        self.data["explained_variance_ratio"] = [
            exp_var / total_var
            for exp_var, total_var in zip(
                self.data["explained_variance"], self.data["total_variance"]
            )
        ]

        # Explained Covariance
        k = self.n_modes
        explained_covariance = []

        # just take the kth column of each transformed view and _compute_covariance
        for i in range(k):
            transformed_views_k = [
                view.isel(mode=slice(i, i + 1)) for view in transformed_views
            ]
            cov_ = self._apply_compute_covariance(
                transformed_views_k, dims_in=["sample", "mode"]
            )
            svals = self._compute_singular_values(cov_, dims_in=["mode1", "mode2"])
            explained_covariance.append(svals.isel(mode=0).item())
        self.data["explained_covariance"] = xr.DataArray(
            explained_covariance, dims=["mode"], coords={"mode": range(1, k + 1)}
        )

        minimum_dimension = min([view[self.feature_name].size for view in views])

        cov = self._apply_compute_covariance(views, dims_in=["sample", "feature"])
        S = self._compute_singular_values(cov, dims_in=["feature1", "feature2"])
        # select every other element starting from the first until the minimum dimension
        self.data["total_explained_covariance"] = (
            S.isel(mode=slice(0, None, 2)).isel(mode=slice(0, minimum_dimension)).sum()
        )
        self.data["explained_covariance_ratio"] = (
            self.data["explained_covariance"] / self.data["total_explained_covariance"]
        )

        return self

    def _compute_singular_values(
        self, x, dims_in=["feature1", "feature2"], dims_out=["mode"]
    ):
        svals = xr.apply_ufunc(
            np.linalg.svd,
            x,
            input_core_dims=[dims_in],
            output_core_dims=[dims_out],
            kwargs={"compute_uv": False},
            vectorize=False,
            dask="allowed",
        )
        svals = svals.assign_coords({"mode": range(1, svals.mode.size + 1)})
        return svals

    def _apply_norm(self, x, dims):
        return xr.apply_ufunc(
            np.linalg.norm,
            x,
            input_core_dims=[dims],
            output_core_dims=[[]],
            kwargs={"axis": -1},
            vectorize=True,
            dask="allowed",
        )

    def _solve_gevp(self, views: Sequence[DataArray], y=None, **kwargs):
        # Setup the eigenvalue problem
        C = self._C(views, dims_in=[self.sample_name, self.feature_name])
        D = self._D(views, **kwargs)
        self.splits = np.cumsum([view.shape[1] for view in views])
        # Solve the eigenvalue problem
        # Get the dimension of _C
        p = C.shape[0]
        subset_by_index = [p - self.n_modes, p - 1]
        # Solve the generalized eigenvalue problem Cx=lambda Dx using a subset of eigenvalues and eigenvectors
        [eigvals, eigvecs] = self._apply_eigh(C, D, subset_by_index=subset_by_index)
        # Sort the eigenvalues and eigenvectors in descending order
        idx_sorted_modes = eigvals.compute().argsort()[::-1]
        idx_sorted_modes = idx_sorted_modes.assign_coords(
            {"mode": range(idx_sorted_modes.mode.size)}
        )
        eigvals = eigvals.isel(mode=idx_sorted_modes)
        eigvecs = eigvecs.isel(mode=idx_sorted_modes).real
        # Set coordiantes
        coords_mode = range(1, eigvals.mode.size + 1)
        coords_feature = C.coords[self.feature_name + "1"].values
        eigvals = eigvals.assign_coords({"mode": coords_mode})
        eigvecs = eigvecs.assign_coords(
            {
                "mode": coords_mode,
                self.feature_name: coords_feature,
            }
        )
        return eigvals, eigvecs

    def _weights(self, eigvals, eigvecs, views, **kwargs):
        # split eigvecs into weights for each view
        # add 0 before the np ndarray splits
        idx = np.concatenate([[0], self.splits])
        self.data["weights"] = [
            eigvecs.isel({self.feature_name: slice(idx[i], idx[i + 1])})
            for i in range(len(idx) - 1)
        ]
        if self.pca:
            # go from weights in PCA space to weights in original space
            n_modes = [data.feature.size for data in self.data["pca_data"]]
            self.data["weights"] = [
                xr.dot(
                    pca.components()
                    .isel(mode=slice(0, n_modes[i]))
                    .rename({"mode": "temp_dim"}),
                    self.data["weights"][i].rename({"feature": "temp_dim"}),
                    dims="temp_dim",
                    optimize=True,
                )
                for i, pca in enumerate(self.pca_models)
            ]

    def _apply_eigh(self, a, b, subset_by_index):
        return xr.apply_ufunc(
            eigh,
            a,
            b,
            input_core_dims=[
                [self.feature_name + "1", self.feature_name + "2"],
                [self.feature_name + "1", self.feature_name + "2"],
            ],
            output_core_dims=[["mode"], ["feature", "mode"]],
            kwargs={"subset_by_index": subset_by_index},
            vectorize=False,
            dask="allowed",
        )

    def _C(self, views, dims_in):
        C = self._apply_compute_covariance(views, dims_in=dims_in)
        return C / len(views)

    def _apply_compute_covariance(
        self, views: Sequence[DataArray], dims_in, dims_out=None
    ) -> DataArray:
        if dims_out is None:
            dims_out = [dims_in[1] + "1", dims_in[1] + "2"]
        all_views = xr.concat(views, dim=dims_in[1])
        C = self._apply_cov(all_views, dims_in=dims_in, dims_out=dims_out)
        Ci = [
            self._apply_cov(view, dims_in=dims_in, dims_out=dims_out) for view in views
        ]
        return C - self._block_diag_dask(Ci, dims_in=dims_out)

    def _apply_cov(
        self, x, dims_in=["sample", "feature"], dims_out=["feature1", "feature2"]
    ):
        if x[dims_in[1]].size == 1:
            return xr.apply_ufunc(
                np.cov,
                x,
                input_core_dims=[dims_in],
                output_core_dims=[[]],
                kwargs={"rowvar": False},
                vectorize=False,
                dask="allowed",
            )
        else:
            C = xr.apply_ufunc(
                np.cov,
                x,
                input_core_dims=[dims_in],
                output_core_dims=[dims_out],
                kwargs={"rowvar": False},
                vectorize=False,
                dask="allowed",
            )
            feature_coords = x.coords[dims_in[1]].values
            C = C.assign_coords(
                {dims_out[0]: feature_coords, dims_out[1]: feature_coords}
            )
            return C

    def _block_diag_dask(self, views, dims_in=["feature1", "featur2"], dims_out=None):
        if dims_out is None:
            dims_out = dims_in
        if all(view.size == 1 for view in views):
            result = da.diag(np.array([view.item() for view in views]))
        else:
            # Extract underlying Dask arrays
            arrays = [da.asarray(view.data) for view in views]

            # Construct a block-diagonal dask array
            blocks = [
                [
                    darr2 if j == i else da.zeros((darr2.shape[0], darr1.shape[0]))
                    for j, darr1 in enumerate(views)
                ]
                for i, darr2 in enumerate(arrays)
            ]

            # Use Dask's block to stack the arrays
            blocked_array = da.block(blocks)

            # Convert the result back to a DataArray
            feature_coords = xr.concat(views, dim=dims_in[0])[dims_in[0]].values
            result = xr.DataArray(
                blocked_array,
                dims=dims_out,
                coords={dims_out[0]: feature_coords, dims_out[1]: feature_coords},
            )
        if any(isinstance(view.data, da.Array) for view in views):
            return result
        else:
            return result.compute()

    def _D(self, views):
        if self.pca:
            blocks = []
            for i, view in enumerate(views):
                pc = self.pca_models[i]
                feature_coords = view.coords[self.feature_name]
                n_features = feature_coords.size
                expvar = pc.explained_variance().isel(mode=slice(0, n_features))
                block = xr.DataArray(
                    da.diag((1 - self.c[i]) * expvar.data + self.c[i]),
                    dims=[self.feature_name + "1", self.feature_name + "2"],
                    coords={
                        self.feature_name + "1": feature_coords.values,
                        self.feature_name + "2": feature_coords.values,
                    },
                )
                block = block.compute()
                blocks.append(block)

        else:
            blocks = [self._apply_E(view, c) for view, c in zip(views, self.c)]

        D = self._block_diag_dask(blocks, dims_in=["feature1", "feature2"])

        D_smallest_eig = self._apply_smallest_eigval(D, dims=["feature1", "feature2"])
        D_smallest_eig = D_smallest_eig - self.eps
        identity_matrix = xr.DataArray(np.eye(D.shape[0]), dims=D.dims, coords=D.coords)
        D = D - D_smallest_eig * identity_matrix
        return D / len(views)

    def _apply_E(self, view, c):
        E = xr.apply_ufunc(
            self._E,
            view,
            input_core_dims=[[self.sample_name, self.feature_name]],
            output_core_dims=[[self.feature_name + "1", self.feature_name + "2"]],
            kwargs={"c": c},
            vectorize=False,
            dask="allowed",
        )
        feature_coords = view.coords[self.feature_name].values
        E = E.assign_coords(
            {
                self.feature_name + "1": feature_coords,
                self.feature_name + "2": feature_coords,
            }
        )
        return E

    def _E(self, view, c):
        return (1 - c) * np.cov(view, rowvar=False) + c * np.eye(view.shape[1])

    def _apply_smallest_eigval(self, D, dims):
        return xr.apply_ufunc(
            self._smallest_eigval,
            D,
            input_core_dims=[dims],
            output_core_dims=[[]],
            vectorize=True,
            dask="allowed",
        )

    def _smallest_eigval(self, D):
        return min(0, np.linalg.eigvalsh(D).min())

    def weights(self) -> List[DataObject]:
        weights = [
            prep.inverse_transform_components(wghts)
            for prep, wghts in zip(self.preprocessors, self.data["weights"])
        ]
        return weights

    def _transform(self, views: Sequence[DataArray]) -> List[DataArray]:
        transformed_views = []
        for i, view in enumerate(views):
            transformed_view = xr.dot(view, self.data["weights"][i], dims="feature")
            transformed_views.append(transformed_view)
        return transformed_views

    def transform(self, views: Sequence[DataObject]) -> List[DataArray]:
        """Transform the input data into the canonical space.

        Parameters
        ----------
        views : List[DataArray | Dataset]
            Input data to transform

        """
        view_preprocessed = []
        for i, view in enumerate(views):
            view_preprocessed = self.preprocessors[i].transform(view)

        transformed_views = self._transform(view_preprocessed)

        unstacked_transformed_views = []
        for i, view in enumerate(transformed_views):
            unstacked_view = self.preprocessors[i].inverse_transform_scores(view)
            unstacked_transformed_views.append(unstacked_view)
        return unstacked_transformed_views

    def components(self, normalize: bool = True) -> List[DataObject]:
        """Get the canonical loadings for each view."""
        can_loads = self.data["canonical_loadings"]
        input_data = self.data["input_data"]
        variates = self.data["variates"]

        if normalize:
            # Compute correlations
            loadings = [
                (
                    loads
                    / data[self.sample_name].size
                    / data.std(self.sample_name)
                    / vari.std(self.sample_name)
                ).clip(-1, 1)
                for loads, data, vari in zip(can_loads, input_data, variates)
            ]
        else:
            loadings = can_loads

        loadings = [
            prep.inverse_transform_components(load)
            for prep, load in zip(self.preprocessors, loadings)
        ]
        return loadings

    def scores(self) -> List[DataArray]:
        """Get the canonical variates for each view."""
        variates = []
        for i, view in enumerate(self.data["variates"]):
            vari = self.preprocessors[i].inverse_transform_scores(view)
            variates.append(vari)
        return variates

    def explained_variance(self) -> List[DataArray]:
        """Get the explained variance for each view."""
        return self.data["explained_variance"]

    def explained_variance_ratio(self) -> List[DataArray]:
        """Get the explained variance ratio for each view."""
        return self.data["explained_variance_ratio"]

    def explained_covariance(self) -> DataArray:
        """Get the explained covariance."""
        return self.data["explained_covariance"]

    def explained_covariance_ratio(self) -> DataArray:
        """Get the explained covariance ratio."""
        return self.data["explained_covariance_ratio"]
