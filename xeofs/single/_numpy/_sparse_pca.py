"""
Sparse PCA via Variable Projection [1]_.

This module provides an implementation of Sparse PCA via Variable Projection,
based heavily on the code from the Ristretto library (https://github.com/erichson/ristretto).

We have made modifications to adapt the algorithm to be used with delayed Dask objects.

This code is licensed under the MIT License. The original library is licensed under the GPL-3.0 License.

References
----------

.. [1] Erichson, N. B. et al. Sparse Principal Component Analysis via Variable Projection. SIAM J. Appl. Math. 80, 977-1002 (2020).

"""

import dask.array.core
import dask.array.linalg
import dask.array.random
import numpy as np
import scipy
from sklearn.utils import check_random_state

from ...utils.data_types import DaskArray


def random_gaussian_map(A, k, random_state):
    """generate random gaussian map

    Parameters
    ----------
    A : array_like, shape `(n, p)`.
        Input array.
    k : integer
        Target rank.
    random_state : RandomState instance
        Random number generator.

    Returns
    -------
    Omega : array_like, `(p, k)`.
        Random gaussian matrix.

    """

    # TODO: adapt for complex-valued data

    if isinstance(A, DaskArray):
        Omega = dask.array.random.standard_normal(
            size=(A.shape[1], k), chunks=(A.chunks[1], -1)
        )
        return Omega.astype(A.dtype)
    else:
        return random_state.standard_normal(size=(A.shape[1], k)).astype(A.dtype)


def johnson_lindenstrauss(A, k, random_state=None):
    """Johnson-Lindenstrauss Random Projection.

    Parameters
    ----------
    A : array_like, shape `(n, p)`.
        Input array.
    k : integer
        Target rank.
    random_state : integer, RandomState instance or None
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    Returns
    -------
    array_like, `(n, k)`.
        Projected matrix.

    """
    # TODO: adapt for complex-valued data
    random_state = check_random_state(random_state)

    if A.ndim != 2:
        raise ValueError("A must be a 2D array, not %dD" % A.ndim)

    # construct gaussian random matrix
    Omega = random_gaussian_map(A, k, random_state)

    # project A onto Omega
    return A.dot(Omega)


def orthonormalize(A, overwrite_a=True, check_finite=False):
    """orthonormalize the columns of A via QR decomposition

    Parameters
    ----------
    A : array_like, shape `(n, p)`.
        Input array.
    overwrite_a : bool, optional
        Whether to overwrite data in A (may improve performance).
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.

    Returns
    -------
    Q : array_like, `(n, p)`.
        Orthonormal basis matrix.

    """
    # TODO: adapt for complex-valued data
    if isinstance(A, DaskArray):
        # NOTE: We expect A  to be tall-and-skinny matrix that is chunked along rows
        # This means our input data is assumed to be chunked along the feature dimensions
        try:
            Q, _ = dask.array.linalg.tsqr(A)
        except ValueError:
            raise ValueError(
                "Data not chunked correctly. Ensure that the data is chunked along the feature dimensions."
            )
    else:
        # NOTE: for A(n, p) 'economic' returns Q(n, k), R(k, p) where k is min(n, p)
        # TODO: when does overwrite_a even work? (fortran?)
        QR = scipy.linalg.qr(
            A,
            overwrite_a=overwrite_a,
            check_finite=check_finite,
            mode="economic",
            pivoting=False,
        )
        Q = QR[0]
    return Q


def perform_subspace_iterations(A, Q, n_iter=2):
    """perform subspace iterations on Q

    Parameters
    ----------
    A : array_like, shape `(n, p)`.
        Input array.
    Q : array_like, shape `(n, k)`.
        Orthonormal basis matrix.
    n_iter : integer, default: 2.
        Number of subspace iterations.

    Returns
    -------
    Q : array_like, `(n, k)`.
        Orthonormal basis matrix.

    """
    # TODO: adapt for complex-valued data
    # orthonormalize Y, overwriting
    Q = orthonormalize(Q)

    # perform subspace iterations
    for _ in range(n_iter):
        Z = orthonormalize(A.T.dot(Q))
        Q = orthonormalize(A.dot(Z))

    return Q


def conjugate_transpose(A):
    """Performs conjugate transpose of A"""
    if np.iscomplexobj(A):
        return A.conj().T
    return A.T


def _compute_rqb(A, rank, oversample=10, n_subspace=2, random_state=None):
    """Randomized QB Decomposition.

    Randomized algorithm for computing the approximate low-rank QB
    decomposition of a rectangular `(n, p)` matrix `A`, with target rank
    `rank << min{n, p}`. The input matrix is factored as `A = Q * B`.

    The quality of the approximation can be controlled via the oversampling
    parameter `oversample` and `n_subspace` which specifies the number of
    subspace iterations.

    Parameters
    ----------
    A : array_like, shape `(n, p)`.
        Input array.
    rank : integer
        Target rank. Best if `rank << min{n, p}`
    oversample : integer, optional (default: 10)
        Controls the oversampling of column space. Increasing this parameter
        may improve numerical accuracy.
    n_subspace : integer, default: 2.
        Parameter to control number of subspace iterations. Increasing this
        parameter may improve numerical accuracy. Every additional subspace
        iterations requires an additional full pass over the data matrix.
    random_state : integer, RandomState instance or None
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    Returns
    -------
    Q:  array_like, shape `(n, rank + oversample)`.
        Orthonormal basis matrix.
    B : array_like, shape `(rank + oversample, p)`.
        Smaller matrix.

    """
    # TODO: adapt for complex-valued data
    Q = johnson_lindenstrauss(A, rank + oversample, random_state=random_state)

    if n_subspace > 0:
        Q = perform_subspace_iterations(A, Q, n_iter=n_subspace)
    else:
        Q = orthonormalize(Q)

    # Project the data matrix a into a lower dimensional subspace
    B = conjugate_transpose(Q).dot(A)

    return Q, B


def compute_rqb(A, rank, oversample=20, n_subspace=2, n_blocks=1, random_state=None):
    """Randomized QB Decomposition.

    Randomized algorithm for computing the approximate low-rank QB
    decomposition of a rectangular `(n, p)` matrix `A`, with target rank
    `rank << min{n, p}`. The input matrix is factored as `A = Q * B`.

    The quality of the approximation can be controlled via the oversampling
    parameter `oversample` and `n_subspace` which specifies the number of
    subspace iterations.

    Parameters
    ----------
    A : array_like, shape `(n, p)`.
        Input array.

    rank : integer
        Target rank. Best if `rank << min{n, p}`

    oversample : integer, optional (default: 10)
        Controls the oversampling of column space. Increasing this parameter
        may improve numerical accuracy.

    n_subspace : integer, default: 2.
        Parameter to control number of subspace iterations. Increasing this
        parameter may improve numerical accuracy. Every additional subspace
        iterations requires an additional full pass over the data matrix.

    n_blocks : integer, default: 1.
        If `n_blocks > 1` a column blocked QB decomposition procedure will be
        performed. A larger number requires less fast memory, while it
        leads to a higher computational time.

    random_state : integer, RandomState instance or None, optional (default `None`)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    Returns
    -------
    Q:  array_like, shape `(n, rank + oversample)`.
        Orthonormal basis matrix.

    B : array_like, shape `(rank + oversample, p)`.
        Smaller matrix.

    References
    ----------
    N. Halko, P. Martinsson, and J. Tropp.
    "Finding structure with randomness: probabilistic
    algorithms for constructing approximate matrix
    decompositions" (2009).
    (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).

    S. Voronin and P.Martinsson.
    "RSVDPACK: Subroutines for computing partial singular value
    decompositions via randomized sampling on single core, multi core,
    and GPU architectures" (2015).
    (available at `arXiv <http://arxiv.org/abs/1502.05366>`_).

    """
    # TODO: adapt for complex-valued data
    if n_blocks > 1:
        if isinstance(A, DaskArray):
            raise ValueError(
                f"For delayed-dask computation n_blocks must be 1, but found n_blocks={n_blocks}."
            )
        n, _ = A.shape

        # index sets
        row_sets = np.array_split(range(n), n_blocks)

        Q_block = []
        K = []

        nblock = 1
        for rows in row_sets:
            # converts A to array, raise ValueError if A has inf or nan
            Qtemp, Ktemp = _compute_rqb(
                A[rows, :],
                rank=rank,
                oversample=oversample,
                n_subspace=n_subspace,
                random_state=random_state,
            )

            Q_block.append(Qtemp)
            K.append(Ktemp)
            nblock += 1

        Q_small, B = _compute_rqb(
            np.concatenate(K, axis=0),
            rank=rank,
            oversample=oversample,
            n_subspace=n_subspace,
            random_state=random_state,
        )

        Q_small = np.vsplit(Q_small, n_blocks)

        Q = [Q_block[i].dot(Q_small[i]) for i in range(n_blocks)]
        Q = np.concatenate(Q, axis=0)

    else:
        Q, B = _compute_rqb(
            A,
            rank=rank,
            oversample=oversample,
            n_subspace=n_subspace,
            random_state=random_state,
        )

    return Q, B


def soft_l0(arr, thresh):
    """Soft threshold operator for l0 regularization.

    Parameters
    ----------
    arr : array_like
        Input array.
    thresh : float
        Threshold value.

    Returns
    -------
    array_like
        Thresholded array.

    """
    # TODO: adapt for complex-valued data
    idx = arr**2 < 2 * thresh
    arr[idx] = 0
    return arr


def soft_l1(arr, thresh):
    """Soft threshold operator for l1 regularization.

    Parameters
    ----------
    arr : array_like
        Input array.
    thresh : float
        Threshold value.

    Returns
    -------
    array_like
        Thresholded array.

    """
    # TODO: adapt for complex-valued data
    return np.sign(arr) * np.maximum(np.abs(arr) - thresh, 0)


def compute_residual(X, B, A):
    # TODO: adapt for complex-valued data
    return X - X.dot(B).dot(A.T)


def compute_spca(
    X,
    n_components=None,
    alpha=0.1,
    beta=1e-5,
    gamma=0.1,
    robust=False,
    regularizer="l1",
    max_iter=1e3,
    tol=1e-5,
    compute=False,
):
    r"""Sparse Principal Component Analysis (SPCA).

    Given a mean centered rectangular matrix `A` with shape `(m, n)`, SPCA
    computes a set of sparse components that can optimally reconstruct the
    input data.  The amount of sparseness is controllable by the coefficient
    of the L1 penalty, given by the parameter alpha. In addition, some ridge
    shrinkage can be applied in order to improve conditioning.


    Parameters
    ----------
    X : array_like, shape `(n, p)`.
        Input array.

    n_components : integer, `n_components << min{m,n}`.
        Target rank, i.e., number of sparse components to be computed.

    alpha : float, (default ``alpha = 0.1``).
        Sparsity controlling parameter. Higher values lead to sparser components.

    beta : float, (default ``beta = 1e-5``).
        Amount of ridge shrinkage to apply in order to improve conditionin.

    regularizer : string {'l0', 'l1'}.
        Type of sparsity-inducing regularizer. The l1 norm (also known as LASSO)
        leads to softhreshold operator (default).  The l0 norm is implemented
        via a hardthreshold operator.

    robust : bool ``{'True', 'False'}``, optional (default ``False``).
        Use a robust algorithm to compute the sparse PCA.

    max_iter : integer, (default ``max_iter = 500``).
        Maximum number of iterations to perform before exiting.

    tol : float, (default ``tol = 1e-5``).
        Stopping tolerance for reconstruction error.

    compute: bool
        Whether to eagerly compute the rotation matrix. If ``True``, then
        the rotation algorithm will check for convergence at each iteration
        and stop once reached. If ``False``, then the rotation algorithm
        can build the matrix lazily, but it will not check for convergence
        and will run all of ``max_iter``.

    Returns
    -------
    B:  array_like, `(p, n_components)`.
        Sparse components extracted from the data.

    A : array_like, `(p, n_components)`.
        Orthogonal components extracted from the data.

    eigvals : array_like, `(n_components)`.
        Eigenvalues correspnding to the extracted components.

    Notes
    -----
    Variable Projection for PCA solves the following optimization problem:
    minimize :math:`1/2 \| X - X B A^T \|^2 + \alpha \|B\|_1 + 1/2 \beta \|B\|^2`
    """
    # TODO: adapt for complex-valued data

    if isinstance(X, DaskArray):

        def svd_algorithm(X):
            return dask.array.linalg.svd(X)
    else:
        # TODO: can we pass the arguments without using functions?
        # linalg.svd(X, full_matrices=False, overwrite_a=False)
        # perhaps we can use Decomposer class to handle this
        def svd_algorithm(X):
            return scipy.linalg.svd(X, full_matrices=False, overwrite_a=False)

    if regularizer == "l1":
        regularizer_func = soft_l1
    elif regularizer == "l0":
        if robust:
            raise NotImplementedError(
                "l0 regularization is not supported for " "robust sparse pca"
            )
        regularizer_func = soft_l0
    else:
        raise ValueError(
            'regularizer must be one of ("l1", "l0"), not ' "%s." % regularizer
        )

    m, n = X.shape
    if n_components is not None:
        if n_components > n:
            raise ValueError(
                "n_components must be less than the number " "of columns of X (%d)" % n
            )
    else:
        n_components = n

    # Initialization of Variable Projection Solver
    U, D, Vt = svd_algorithm(X)
    Dmax = D[0]  # l2 norm

    A = Vt[:n_components].T
    B = Vt[:n_components].T

    if robust:
        U = U[:, :n_components]
        Vt = Vt[:n_components]
        S = np.zeros_like(X)
    else:
        # compute outside the loop
        VD = Vt.T * D
        VD2 = Vt.T * D**2

    # Set Tuning Parameters
    alpha *= Dmax**2
    beta *= Dmax**2
    nu = 1.0 / (Dmax**2 + beta)
    kappa = nu * alpha

    obj = []  # values of objective function
    n_iter = 0

    #   Apply Variable Projection Solver
    while max_iter > n_iter:
        # Update A:
        # X'XB = UDV'
        # Compute X'XB via SVD of X
        if robust:
            XS = X - S
            XB = X.dot(B)
            Z = (XS).T.dot(XB)
        else:
            Z = VD2.dot(Vt.dot(B))

        Utilde, Dtilde, Vttilde = svd_algorithm(Z)
        A = Utilde.dot(Vttilde)

        # Proximal Gradient Descent to Update B
        if robust:
            R = XS - XB.dot(A.T)
            G = X.T.dot(R.dot(A)) - beta * B
        else:
            G = VD2.dot(Vt.dot(A - B)) - beta * B

        B = regularizer_func(B + nu * G, kappa)

        if robust:
            R = compute_residual(X, B, A)
            S = soft_l1(R, gamma)
            R -= S
        else:
            R = compute_residual(VD.T, B, A)

        objective = (
            0.5 * np.sum(R**2) + alpha * np.sum(np.abs(B)) + 0.5 * beta * np.sum(B**2)
        )
        if robust:
            objective += gamma * np.sum(np.abs(S))

        obj.append(objective)

        # Break if obj is not improving anymore
        if compute and n_iter > 0 and abs(obj[-2] - obj[-1]) / obj[-1] < tol:
            break

        # Next iter
        n_iter += 1

    eigen_values = Dtilde / (m - 1)

    return B, A, eigen_values


def compute_rspca(
    X,
    n_components,
    alpha=0.1,
    beta=0.1,
    max_iter=1e3,
    regularizer="l1",
    tol=1e-5,
    oversample=50,
    n_subspace=2,
    n_blocks=1,
    robust=False,
    random_state=None,
    compute=False,
):
    r"""Randomized Sparse Principal Component Analysis (rSPCA).

    Given a mean centered rectangular matrix `A` with shape `(n, p)`, SPCA
    computes a set of sparse components that can optimally reconstruct the
    input data.  The amount of sparseness is controllable by the coefficient
    of the L1 penalty, given by the parameter alpha. In addition, some ridge
    shrinkage can be applied in order to improve conditioning.

    This algorithm uses randomized methods for linear algebra to accelerate
    the computations.

    The quality of the approximation can be controlled via the oversampling
    parameter `oversample` and `n_subspace` which specifies the number of
    subspace iterations.


    Parameters
    ----------
    X : array_like, shape `(n, p)`.
        Real nonnegative input matrix.

    n_components : integer, `n_components << min{m,n}`.
        Target rank, i.e., number of sparse components to be computed.

    alpha : float, (default ``alpha = 0.1``).
        Sparsity controlling parameter. Higher values lead to sparser components.

    beta : float, (default ``beta = 0.1``).
        Amount of ridge shrinkage to apply in order to improve conditionin.

    regularizer : string {'l0', 'l1'}.
        Type of sparsity-inducing regularizer. The l1 norm (also known as LASSO)
        leads to softhreshold operator (default).  The l0 norm is implemented
        via a hardthreshold operator.

    max_iter : integer, (default ``max_iter = 500``).
        Maximum number of iterations to perform before exiting.

    tol : float, (default ``tol = 1e-5``).
        Stopping tolerance for reconstruction error.

    oversample : integer, optional (default: 10)
        Controls the oversampling of column space. Increasing this parameter
        may improve numerical accuracy.

    n_subspace : integer, default: 2.
        Parameter to control number of subspace iterations. Increasing this
        parameter may improve numerical accuracy.

    n_blocks : integer, default: 2.
        Paramter to control in how many blocks of columns the input matrix
        should be split. A larger number requires less fast memory, while it
        leads to a higher computational time.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    Returns
    -------
    B:  array_like, `(p, n_components)`.
        Sparse components extracted from the data.

    A : array_like, `(p, n_components)`.
        Orthogonal components extracted from the data.

    eigvals : array_like, `(n_components)`.
        Eigenvalues correspnding to the extracted components.

    S : array_like, `(n, p)`.
        Sparse component which captures grossly corrupted entries in the data
        matrix. Returned only if `robust == True`

    obj : array_like, `(n_iter)`.
        Objective value at the i-th iteration.

    Notes
    -----
    Variable Projection for SPCA solves the following optimization problem:
    minimize :math:`1/2 \| X - X B A^T \|^2 + \alpha \|B\|_1 + 1/2 \beta \|B\|^2`
    """

    # TODO: adapt for complex-valued data

    # Shape of data matrix
    m = X.shape[0]

    # Compute QB decomposition
    _, Xcompressed = compute_rqb(
        X,
        rank=n_components,
        oversample=oversample,
        n_subspace=n_subspace,
        n_blocks=n_blocks,
        random_state=random_state,
    )

    # Compute Sparse PCA
    B, A, eigen_values = compute_spca(
        Xcompressed,
        n_components=n_components,
        alpha=alpha,
        beta=beta,
        regularizer=regularizer,
        max_iter=max_iter,
        tol=tol,
        robust=robust,
        compute=compute,
    )
    # rescale eigen values
    eigen_values *= (n_components + oversample - 1) / (m - 1)

    return B, A, eigen_values
