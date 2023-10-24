import numpy as np
import numba

VALID_KERNELS = ["bisquare", "gaussian", "exponential"]


@numba.njit(fastmath=True)
def kernel_weights_nb(distance, bandwidth, kernel):
    if kernel == "bisquare":
        return _bisquare_nb(distance, bandwidth)
    elif kernel == "gaussian":
        return _gaussian_nb(distance, bandwidth)
    elif kernel == "exponential":
        return _exponential_nb(distance, bandwidth)
    else:
        raise ValueError(
            f"Invalid kernel: {kernel}. Must be one of ['bisquare', 'gaussian', 'exponential']."
        )


@numba.njit(fastmath=True)
def _bisquare_nb(distance, bandwidth):
    weights = (1 - (distance / bandwidth) ** 2) ** 2
    return np.where(distance <= bandwidth, weights, 0)


@numba.njit(fastmath=True)
def _gaussian_nb(distance, bandwidth):
    return np.exp(-0.5 * (distance / bandwidth) ** 2)


@numba.njit(fastmath=True)
def _exponential_nb(distance, bandwidth):
    return np.exp(-0.5 * (distance / bandwidth))
