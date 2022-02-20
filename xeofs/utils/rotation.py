''' Implementation of VARIMAX and PROMAX rotation. '''

# =============================================================================
# Imports
# =============================================================================
import numpy as np


# =============================================================================
# VARIMAX
# =============================================================================
def varimax(
    X : np.ndarray,
    gamma : float = 1,
    max_iter : int = 1000,
    rtol : float = 1e-8
):
    '''
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

    Returns
    -------
    Xrot : np.ndarray
        Rotated matrix with same dimensions as X.
    R : array-like
        Rotation matrix of shape ``(n_rot x n_rot)``

    '''
    X = X.copy()
    n_samples, n_modes = X.shape

    if n_modes < 2:
        err_msg = 'Cannot rotate {:} modes (columns), but must be 2 or more.'
        err_msg = err_msg.format(n_modes)
        raise ValueError(err_msg)

    # Initialize rotation matrix
    R = np.eye(n_modes)

    # Normalize the matrix using square root of the sum of squares (Kaiser)
    h = np.sqrt(np.sum(X * X.conjugate(), axis=1))
    # A = np.diag(1./h) @ A
    X = (1. / h)[:, np.newaxis] * X

    # Seek for rotation matrix based on varimax criteria
    delta = 0.
    converged = False
    for i in range(max_iter):
        delta_old = delta
        basis = X @ R

        basis2 = basis * basis.conjugate()
        basis3 = basis2 * basis
        W = np.diag(np.sum(basis2, axis=0))
        alpha = gamma / n_samples

        transformed = X.conjugate().T @ (basis3 - (alpha * basis @ W))
        U, svals, VT = np.linalg.svd(transformed)
        R = U @ VT
        delta = np.sum(svals)
        if (abs(delta - delta_old) / delta) < rtol:
            converged = True
            break

    if(not converged):
        raise RuntimeError('Rotation process did not converge.')

    # De-normalize
    X = h[:, np.newaxis] * X

    # Rotate
    Xrot = X @ R
    return Xrot, R


# =============================================================================
# PROMAX
# =============================================================================
def promax(
    X : np.ndarray,
    power : int = 1,
    max_iter : int = 1000,
    rtol : float = 1e-8
):
    '''
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

    '''
    X = X.copy()

    # Perform varimax rotation
    X, rot_mat = varimax(X=X, max_iter=max_iter, rtol=rtol)

    # Pre-normalization by communalities (sum of squared rows)
    h = np.sqrt(np.sum(X * X.conjugate(), axis=1))
    X = (1. / h)[:, np.newaxis] * X

    # Max-normalisation of columns
    Xnorm = X / np.max(abs(X), axis=0)

    # "Procustes" equation
    P = Xnorm * np.abs(Xnorm)**(power - 1)

    # Fit linear regression model of "Procrustes" equation
    # see Richman 1986 for derivation
    L = np.linalg.inv(X.conjugate().T @ X) @ X.conjugate().T @ P

    # calculate diagonal of inverse square
    try:
        sigma_inv = np.diag(np.diag(np.linalg.inv(L.conjugate().T @ L)))
    except np.linalg.LinAlgError:
        sigma_inv = np.diag(np.diag(np.linalg.pinv(L.conjugate().T @ L)))

    # transform and calculate inner products
    L = L @ np.sqrt(sigma_inv)
    Xrot = X @ L

    # Post-normalization based on Kaiser
    Xrot = h[:, np.newaxis] * Xrot

    rot_mat = rot_mat @ L

    # Correlation matrix
    L_inv = np.linalg.inv(L)
    phi = L_inv @ L_inv.conjugate().T

    return Xrot, rot_mat, phi
