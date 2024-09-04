import numpy as np

from ._svd import _SVD


def _fractional_matrix_power(C, power, **kwargs):
    """Compute the fractional matrix power of a symmetric matrix using SVD.

    Note: This function is a simplified version of the fractional_matrix_power
    function from the scipy library. However, the scipy function does not
    support dask arrays due to the use of np.asarray.
    """
    if C.shape[0] != C.shape[1]:
        raise ValueError("Matrix must be square.")

    svd = _SVD(n_modes="all", **kwargs)
    _, s, V = svd.fit_transform(C)

    # cut off small singular values
    is_above_zero = s > np.finfo(s.dtype).eps
    V = V[:, is_above_zero]
    s = s[is_above_zero]

    # TODO: use hermitian=True for numpy>=2.0
    # V, s, _ = np.linalg.svd(C, hermitian=True)
    C_scaled = V @ np.diag(s**power) @ V.conj().T

    # Even if the input matrix is real and symmetric, the output matrix might
    # be complex due to numerical errors. In this case, we return the real part
    if np.iscomplexobj(C):
        return C_scaled
    else:
        return C_scaled.real
