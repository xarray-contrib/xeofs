from typing import Sequence, Hashable, Optional, Callable
from typing_extensions import Self

from sklearn.utils.extmath import randomized_svd

from xeofs.utils.data_types import DataArray
from xeofs.utils.data_types import Data
from ._base_model import _BaseModel
from ..utils.sanity_checks import validate_input_type
from ..utils.xarray_utils import convert_to_dim_type
from ..utils.constants import (
    VALID_CARTESIAN_X_NAMES,
    VALID_CARTESIAN_Y_NAMES,
    VALID_LATITUDE_NAMES,
    VALID_LONGITUDE_NAMES,
)
from .eof import EOF
import numpy as np
import xarray as xr

from tqdm import trange
import numba
from numba import prange
from ..utils.distance_metrics import distance_nb, VALID_METRICS
from ..utils.kernels import kernel_weights_nb, VALID_KERNELS


class GWPCA(_BaseModel):
    """Geographically weighted PCA.

    Geographically weighted PCA (GWPCA) [1]_ uses a geographically weighted approach to perform PCA for
    each observation in the dataset based on its local neighbors.

    The neighbors for each observation are determined based on the provided
    bandwidth and metric. Each neighbor is weighted based on its distance from
    the observation using the provided kernel function.

    Parameters
    ----------
    n_modes: int
        Number of modes to calculate.
    bandwidth: float
        Bandwidth of the kernel function. Must be > 0.
    metric: str, default="haversine"
        Distance metric to use. Great circle distance (`haversine`) is always expressed in kilometers.
        All other distance metrics are reported in the unit of the input data.
        See scipy.spatial.distance.cdist for a list of available metrics.
    kernel: str, default="bisquare"
        Kernel function to use. Must be one of ['bisquare', 'gaussian', 'exponential'].
    center: bool, default=True
        If True, the data is centered by subtracting the mean (feature-wise).
    standardize: bool, default=False
        If True, the data is divided by the standard deviation (feature-wise).
    use_coslat: bool, default=False
        If True, the data is weighted by the square root of cosine of latitudes.
    sample_name: str, default="sample"
        Name of the sample dimension.
    feature_name: str, default="feature"
        Name of the feature dimension.

    Attributes
    ----------
    bandwidth: float
        Bandwidth of the kernel function.
    metric: str
        Distance metric to use.
    kernel: str
        Kernel function to use.

    Methods:
    --------
    fit(X) : Fit the model with input data.

    explained_variance() : Return the explained variance of the local components.

    explained_variance_ratio() : Return the explained variance ratio of the local components.

    largest_locally_weighted_components() : Return the largest locally weighted components.


    Notes
    -----
    GWPCA is computationally expensive since it performs PCA for each sample. This implementation leverages
    `numba` to speed up the computation on CPUs. However, for moderate to large datasets, this won't be sufficient.
    Currently, GPU support is not implemented. If you're dataset is too large to be processed on a CPU, consider
    using the R package `GWmodel` [2]_, which provides a GPU implementation of GWPCA.

    References
    ----------
    .. [1] Harris, P., Brunsdon, C. & Charlton, M. Geographically weighted principal components analysis. International Journal of Geographical Information Science 25, 1717–1736 (2011).
    .. [2] https://cran.r-project.org/web/packages/GWmodel/index.html


    """

    def __init__(
        self,
        n_modes: int,
        bandwidth: float,
        metric: str = "haversine",
        kernel: str = "bisquare",
        center: bool = True,
        standardize: bool = False,
        use_coslat: bool = False,
        sample_name: str = "sample",
        feature_name: str = "feature",
    ):
        super().__init__(
            n_modes,
            center=center,
            standardize=standardize,
            use_coslat=use_coslat,
            sample_name=sample_name,
            feature_name=feature_name,
        )

        self.attrs.update({"model": "GWPCA"})

        if not kernel in VALID_KERNELS:
            raise ValueError(
                f"Invalid kernel: {kernel}. Must be one of {VALID_KERNELS}."
            )

        if not metric in VALID_METRICS:
            raise ValueError(
                f"Invalid metric: {metric}. Must be one of {VALID_METRICS}."
            )

        if bandwidth <= 0:
            raise ValueError(f"Invalid bandwidth: {bandwidth}. Must be > 0.")

        self.bandwidth = bandwidth
        self.metric = metric
        self.kernel = kernel

    def _fit_algorithm(self, X: DataArray) -> Self:
        # Convert Dask arrays
        if not isinstance(X.data, np.ndarray):
            print(
                "Warning: GWPCA currently does not support Dask arrays. Data is being loaded into memory."
            )
            X = X.compute()
        # 1. Get sample coordinates
        valid_x_names = VALID_CARTESIAN_X_NAMES + VALID_LONGITUDE_NAMES
        valid_y_names = VALID_CARTESIAN_Y_NAMES + VALID_LATITUDE_NAMES
        n_sample_dims = len(self.sample_dims)
        if n_sample_dims == 1:
            indexes = self.preprocessor.preconverter.transformers[0].original_indexes
            sample_dims = self.preprocessor.renamer.transformers[0].sample_dims_after
            xy = None
            for dim in sample_dims:
                keys = [k for k in indexes[dim].coords.keys()]
                x_found = any([k.lower() in valid_x_names for k in keys])
                y_found = any([k.lower() in valid_y_names for k in keys])
                if x_found and y_found:
                    xy = np.asarray([*indexes[dim].values])
                    break
            if xy is None:
                raise ValueError("Cannot find sample coordinates.")
        elif n_sample_dims == 2:
            indexes = self.preprocessor.postconverter.transformers[0].original_indexes
            xy = np.asarray([*indexes[self.sample_name].values])

        else:
            raise ValueError(
                "GWPCA requires number of sample dimensions to be <= 2, but got {n_sample_dims}."
            )

        # 2. Remove NaN samples from sample indexes
        is_no_nan_sample = self.preprocessor.sanitizer.transformers[0].is_valid_sample
        xy = xr.DataArray(
            xy,
            dims=[self.sample_name, "xy"],
            coords={
                self.sample_name: is_no_nan_sample[self.sample_name],
                "xy": ["x", "y"],
            },
            name="index",
        )

        xy = xy[is_no_nan_sample]

        # Iterate over all samples
        kwargs = {
            "n_modes": self.n_modes,
            "metric": self.metric,
            "kernel": self.kernel,
            "bandwidth": self.bandwidth,
        }
        components, exp_var, tot_var = xr.apply_ufunc(
            _local_pcas,
            X,
            xy,
            input_core_dims=[
                [self.sample_name, self.feature_name],
                [self.sample_name, "xy"],
            ],
            output_core_dims=[
                [self.sample_name, self.feature_name, "mode"],
                [self.sample_name, "mode"],
                [self.sample_name],
            ],
            kwargs=kwargs,
            dask="forbidden",
        )
        components = components.assign_coords(
            {
                self.sample_name: X[self.sample_name],
                self.feature_name: X[self.feature_name],
                "mode": np.arange(1, self.n_modes + 1),
            }
        )
        exp_var = exp_var.assign_coords(
            {
                self.sample_name: X[self.sample_name],
                "mode": np.arange(1, self.n_modes + 1),
            }
        )
        tot_var = tot_var.assign_coords({self.sample_name: X[self.sample_name]})

        exp_var_ratio = exp_var / tot_var

        # self.data.add(X, "input_data")
        self.data.add(components, "components")
        self.data.add(exp_var, "explained_variance")
        self.data.add(exp_var_ratio, "explained_variance_ratio")

        self.data.set_attrs(self.attrs)

        return self

    def explained_variance(self):
        expvar = self.data["explained_variance"]
        return self.preprocessor.inverse_transform_scores(expvar)

    def explained_variance_ratio(self):
        expvar = self.data["explained_variance_ratio"]
        return self.preprocessor.inverse_transform_scores(expvar)

    def largest_locally_weighted_components(self):
        comps = self.data["components"]
        idx_max = abs(comps).argmax(self.feature_name)
        input_features = self.preprocessor.stacker.transformers[0].coords_out["feature"]
        llwc = input_features[idx_max].drop_vars(self.feature_name)
        llwc.name = "largest_locally_weighted_components"
        return self.preprocessor.inverse_transform_scores(llwc)

    def scores(self):
        raise NotImplementedError("GWPCA does not support scores() yet.")

    def _transform_algorithm(self, data: DataArray) -> DataArray:
        raise NotImplementedError("GWPCA does not support transform() yet.")

    def _inverse_transform_algorithm(self, data: DataArray) -> DataArray:
        raise NotImplementedError("GWPCA does not support inverse_transform() yet.")


# Additional utility functions for local PCA
# =============================================================================


@numba.njit(fastmath=True, parallel=True)
def _local_pcas(X, xy, n_modes, metric, kernel, bandwidth):
    """Perform local PCA on each sample.

    Parameters
    ----------
    X: ndarray
        Input data with shape (n_samples, n_features)
    xy: ndarray
        Sample coordinates with shape (n_samples, 2)
    n_modes: int
        Number of modes to calculate.
    metric: str
        Distance metric to use. Great circle distance (`haversine`) is always expressed in kilometers.
        All other distance metrics are reported in the unit of the input data.
        See scipy.spatial.distance.cdist for a list of available metrics.
    kernel: str
        Kernel function to use. Must be one of ['bisquare', 'gaussian', 'exponential'].
    bandwidth: float
        Bandwidth of the kernel function.

    Returns
    -------
    ndarray
        Array of local components with shape (n_samples, n_features, n_modes)
    ndarray
        Array of local explained variance with shape (n_samples, n_modes)
    ndarray
        Array of total variance with shape (n_samples,)

    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    Vs = np.empty((n_samples, n_features, n_modes))
    exp_var = np.empty((n_samples, n_modes))
    tot_var = np.empty(n_samples)
    for i in prange(n_samples):
        dist = distance_nb(xy, xy[i], metric=metric)
        weights = kernel_weights_nb(dist, bandwidth, kernel)
        valid_data = weights > 0

        weights = weights[valid_data]
        x = X[valid_data]

        wmean = _wmean_axis0(x, weights)
        x -= wmean

        sqrt_weights = np.sqrt(weights)
        x = _weigh_columns(x, sqrt_weights)

        Ui, si, ViT = np.linalg.svd(x, full_matrices=False)
        # Renormalize singular values
        si = si**2 / weights.sum()
        ti = si.sum()

        si = si[:n_modes]
        ViT = ViT[:n_modes]
        Vi = ViT.T

        Vs[i] = Vi
        exp_var[i, : len(si)] = si
        tot_var[i] = ti

    return Vs, exp_var, tot_var


@numba.njit(fastmath=True)
def _wmean_axis0(X, weights):
    """Compute weighted mean along axis 0.

    Numba version of np.average. Note that np.average is supported by Numba,
    but is restricted to `X` and `weights` having the same shape.
    """
    wmean = np.empty(X.shape[1])
    for i in prange(X.shape[1]):
        wmean[i] = np.average(X[:, i], weights=weights)
    return wmean


@numba.njit(fastmath=True)
def _weigh_columns(x, weights):
    """Weigh columns of x by weights.

    Numba version of broadcasting.

    Parameters
    ----------
    x: ndarray
        Input data with shape (n_samples, n_features)
    weights: ndarray
        Weights with shape (n_samples,)

    Returns
    -------
    x_weighted: ndarray
        Weighted data with shape (n_samples, n_features)
    """
    x_weighted = np.zeros_like(x)
    for i in range(x.shape[1]):
        x_weighted[:, i] = x[:, i] * weights
    return x_weighted


@numba.guvectorize(
    [
        (
            numba.float32[:, :],
            numba.float32[:, :],
            numba.float32[:],
            numba.int32[:],
            numba.int32,
            numba.float32,
            numba.int32,
            numba.float32[:, :],
            numba.float32[:],
            numba.float32,
        )
    ],
    # In order to specify the output dimension which has not been defined in the input dimensions
    # one has to use a dummy variable (see Numba #2797 https://github.com/numba/numba/issues/2797)
    "(n,m),(n,o),(o),(n_out),(),(),()->(m,n_out),(n_out),()",
)
def local_pca_vectorized(
    data, XY, xy, n_out, metric, bandwidth, kernel, comps, expvar, totvar
):
    """Perform local PCA

    Numba vectorized version of local_pca.

    Parameters
    ----------
    data: ndarray
        Input data with shape (n_samples, n_features)
    XY: ndarray
        Sample coordinates with shape (n_samples, 2)
    xy: ndarray
        Coordinates of the sample to perform PCA on with shape (2,)
    n_out: ndarray
        Number of modes to calculate. (see comment above; workaround for Numba #2797)
    metric: int
        Numba only accepts int/floats; so metric str has to be converted first e.g. by a simple dictionary (not implemented yet)
        see Numba #4404 (https://github.com/numba/numba/issues/4404)
    bandwidth: float
        Bandwidth of the kernel function.
    kernel: int
        Numba only accepts int/floats; so kernel str has to be converted first e.g. by a simple dictionary (not implemented yet)
        see Numba #4404 (https://github.com/numba/numba/issues/4404)
    comps: ndarray
        Array of local components with shape (n_features, n_modes)
    expvar: ndarray
        Array of local explained variance with shape (n_modes)
    totvar: ndarray
        Array of total variance with shape (1)


    """
    distance = distance_nb(XY, xy, metric=metric)
    weights = kernel_weights_nb(distance, bandwidth, kernel)
    is_positive_weight = weights > 0
    X = data[is_positive_weight]
    weights = weights[is_positive_weight]

    wmean = _wmean_axis0(X, weights)
    X -= wmean

    sqrt_weights = np.sqrt(weights)
    X = _weigh_columns(X, sqrt_weights)

    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    Vt = Vt[: n_out.shape[0], :]
    lbda = s**2 / weights.sum()
    for i in range(n_out.shape[0]):
        expvar[i] = lbda[i]
        comps[:, i] = Vt[i, :]
    totvar = lbda.sum()
