import numpy as np
import numba
from numba import prange
from scipy.spatial.distance import cdist

from .constants import AVG_EARTH_RADIUS

VALID_METRICS = ["euclidean", "haversine"]


def distance_matrix_bc(A, B, metric="haversine"):
    """Compute a distance matrix between two arrays using broadcasting.

    Parameters
    ----------
    A: 2D darray
        Array of longitudes and latitudes with shape (N, 2)
    B: 2D darray
        Array of longitudes and latitudes with shape (M, 2)
    metric: str
        Distance metric to use. Great circle distance (`haversine`) is always expressed in kilometers.
        All other distance metrics are reported in the unit of the input data.
        See scipy.spatial.distance.cdist for a list of available metrics.

    Returns
    -------
    distance: 2D darray
        Distance matrix with shape (N, M)


    """
    if metric == "haversine":
        return _haversine_distance_bc(A, B)
    else:
        return cdist(XA=A, XB=B, metric=metric)


def _haversine_distance_bc(lonlats1, lonlats2):
    """Compute the great circle distance matrix between two arrays

    This implementation uses numpy broadcasting.

    Parameters
    ----------
    lonlats1: 2D darray
        Array of longitudes and latitudes with shape (N, 2)
    lonlats2: 2D darray
        Array of longitudes and latitudes with shape (M, 2)

    Returns
    -------
    distance: 2D darray
        Great circle distance matrix with shape (N, M) in kilometers

    """
    # Convert to radians
    lonlats1 = np.radians(lonlats1)
    lonlats2 = np.radians(lonlats2)

    # Extract longitudes and latitudes
    lon1, lat1 = lonlats1[:, 0], lonlats1[:, 1]
    lon2, lat2 = lonlats2[:, 0], lonlats2[:, 1]

    # Compute differences in longitudes and latitudes
    dlon = lon2 - lon1[:, np.newaxis]
    dlat = lat2 - lat1[:, np.newaxis]

    # Haversine formula
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)[..., None] * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = AVG_EARTH_RADIUS * c

    return distance


@numba.njit(fastmath=True)
def distance_nb(A, b, metric="euclidean"):
    if metric == "euclidean":
        return _euclidian_distance_nb(A, b)
    elif metric == "haversine":
        return _haversine_distance_nb(A, b)
    else:
        raise ValueError(
            f"Invalid metric: {metric}. Must be one of ['euclidean', 'haversine']."
        )


@numba.njit(fastmath=True)
def _euclidian_distance_nb(A, b):
    """Compute the Euclidian distance between two arrays.

    This implementation uses numba.

    Parameters
    ----------
    A: 2D array
        Array of shape (N, P)
    b: 1D array
        Array of shape (P,)

    Returns
    -------
    distance: 1D array
        Distance matrix with shape (N,)

    """
    dist = np.zeros(A.shape[0])
    for r in prange(A.shape[0]):
        d = 0
        for c in range(A.shape[1]):
            d += (b[c] - A[r, c]) ** 2
        dist[r] = d
    return np.sqrt(dist)


@numba.njit(fastmath=True)
def _haversine_distance_nb(A, b):
    # Convert to radians
    A = np.radians(A)
    b = np.radians(b)

    # Compute differences in longitudes and latitudes
    dlon = b[0] - A[:, 0]
    dlat = b[1] - A[:, 1]

    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(A[:, 1]) * np.cos(b[1]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = AVG_EARTH_RADIUS * c

    return distance
