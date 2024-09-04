import numpy as np
from dask.array import Array as DaskArray  # type: ignore
from dask.array.linalg import svd_compressed


def _promax(
    X: np.ndarray,
    power: int = 1,
    max_iter: int = 1000,
    rtol: float = 1e-8,
    compute: bool = True,
):
    """
    Perform (oblique) Promax rotation.

    This implementation also works for complex numbers.

    Parameters
    ----------
    X : np.ndarray
        2D matrix to be rotated. Must have shape ``p x m`` containing
        p features and m modes.
    power : int
        Rotation parameter defining the power the Varimax solution is raised
        to. For ``power=1``, this is equivalent to the Varimax solution
        (the default is 1).
    max_iter: int
        Maximum number of iterations for finding the rotation matrix
        (the default is 1000).
    rtol:
        The relative tolerance for the rotation process to achieve
        (the default is 1e-8).
    compute: bool
        Whether to eagerly compute the rotation matrix. If ``True``, then
        the rotation algorithm will check for convergence at each iteration
        and stop once reached. If ``False``, then the rotation algorithm
        can build the matrix lazily, but it will not check for convergence
        and will run all of ``max_iter``.

    Returns
    -------
    Xrot : np.ndarray
        2D matrix containing the rotated modes.
    rot_mat : np.ndarray
        Rotation matrix of shape ``m x m`` with m being number of modes.
    phi : np.ndarray
        Correlation matrix of PCs of shape ``m x m`` with m being number
        of modes. For Varimax solution (``power=1``), the correlation matrix
        is diagonal i.e. the modes are uncorrelated.

    """
    X = X.copy()

    # Perform varimax rotation
    X, rot_mat = _varimax(X=X, max_iter=max_iter, rtol=rtol, compute=compute)

    # Pre-normalization by communalities (sum of squared rows)
    h = np.sqrt(np.sum(X * X.conj(), axis=1))
    # Add a stabilizer to avoid zero communalities
    eps = np.finfo(X.dtype).eps
    X = (1.0 / (h + eps))[:, np.newaxis] * X

    # Max-normalisation of columns
    Xnorm = X / np.max(abs(X), axis=0)

    # "Procustes" equation
    P = Xnorm * np.abs(Xnorm) ** (power - 1)

    # Fit linear regression model of "Procrustes" equation
    # see Richman 1986 for derivation
    L = np.linalg.inv(X.conj().T @ X) @ X.conj().T @ P

    # calculate diagonal of inverse square
    try:
        sigma_inv = np.diag(np.diag(np.linalg.inv(L.conj().T @ L)))
    except np.linalg.LinAlgError:
        sigma_inv = np.diag(np.diag(np.linalg.pinv(L.conj().T @ L)))

    # transform and calculate inner products
    L = L @ np.sqrt(sigma_inv)
    Xrot = X @ L

    # Post-normalization based on Kaiser
    Xrot = h[:, np.newaxis] * Xrot

    rot_mat = rot_mat @ L

    # Correlation matrix
    L_inv = np.linalg.inv(L)
    phi = L_inv @ L_inv.conj().T

    return Xrot, rot_mat, phi


def _varimax(
    X: np.ndarray,
    gamma: float = 1,
    max_iter: int = 1000,
    rtol: float = 1e-8,
    compute: bool = True,
):
    """
    Perform (orthogonal) Varimax rotation.

    This implementation also works for complex numbers.

    Parameters
    ----------
    X : np.ndarray
        2D matrix to be rotated containing features as rows and modes as
        columns.
    gamma : float
        Parameter which determines the type of rotation performed: varimax (1),
        quartimax (0). Other values are possible. The default is 1.
    max_iter : int
        Number of iterations performed. The default is 1000.
    rtol : float
        Relative tolerance at which iteration process terminates.
        The default is 1e-8.
    compute: bool
        Whether to eagerly compute the rotation matrix. If ``True``, then
        the rotation algorithm will check for convergence at each iteration
        and stop once reached. If ``False``, then the rotation algorithm
        can build the matrix lazily, but it will not check for convergence
        and will run all of ``max_iter``.

    Returns
    -------
    Xrot : np.ndarray
        Rotated matrix with same dimensions as X.
    R : array-like
        Rotation matrix of shape ``(n_rot x n_rot)``

    """
    X = X.copy()
    n_samples, n_modes = X.shape

    if isinstance(X, DaskArray):
        # Use svd_compressed if dask to allow chunking in both dimensions
        svd_func = svd_compressed
        svd_args = (n_modes,)
    else:
        svd_func = np.linalg.svd
        svd_args = ()

    if n_modes < 2:
        err_msg = "Cannot rotate {:} modes (columns), but must be 2 or more."
        err_msg = err_msg.format(n_modes)
        raise ValueError(err_msg)

    # Initialize rotation matrix
    R = np.eye(n_modes)

    # Normalize the matrix using square root of the sum of squares (Kaiser)
    h = np.sqrt(np.sum(X * X.conj(), axis=1))
    # A = np.diag(1./h) @ A

    # Add a stabilizer to avoid zero communalities
    eps = np.finfo(X.dtype).eps
    X = (1.0 / (h + eps))[:, np.newaxis] * X

    # Seek for rotation matrix based on varimax criteria
    delta = 0.0
    for i in range(max_iter):
        delta_old = delta
        basis = X @ R

        basis2 = basis * basis.conj()
        basis3 = basis2 * basis
        W = np.diag(np.sum(basis2, axis=0))
        alpha = gamma / n_samples

        transformed = X.conj().T @ (basis3 - (alpha * basis @ W))
        U, svals, VT = svd_func(transformed, *svd_args)
        R = U @ VT
        delta = np.sum(svals)
        if compute and (abs(delta - delta_old) / delta) < rtol:
            break

    if compute and (abs(delta - delta_old) / delta) > rtol:
        raise RuntimeError("Rotation process did not converge.")

    # De-normalize
    X = h[:, np.newaxis] * X

    # Rotate
    Xrot = X @ R
    return Xrot, R
