import numpy as np

import numba
from numba import prange
from ..utils.distance_metrics import distance_nb
from ..utils.kernels import kernel_weights_nb


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
