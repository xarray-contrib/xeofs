'''Distance matrix between two sets of time series.

The original code was written by D. Bueso and was taken from [2]_.

Original reference:
- Bueso, D., Piles, M. & Camps-Valls, G. Nonlinear PCA for Spatio-Temporal
Analysis of Earth Observation Data. IEEE Trans. Geosci. Remote Sensing 1–12
(2020) doi:10.1109/TGRS.2020.2969813.

References
----------
.. [1] https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8989964&casa_token=3zKG0dtp-ewAAAAA:FM1CrVISSSqhWEAwPGpQqCgDYccfLG4N-67xNNDzUBQmMvtIOHuC7T6X-TVQgbDg3aDOpKBksg&tag=1
.. [2] https://github.com/DiegoBueso/ROCK-PCA

'''
import numpy as np


def distance_matrix(X1, X2, algorithm='original'):
    D = - 2 * (X1.conj().T @ X2)
    if algorithm == 'original':
        D = D + np.sum(X1 * X1, axis=0)[..., None]
        D = D + np.sum(X2 * X2, axis=0)
    # NOTE: Is this implementation really correct?
    # The distance matrix is not zero on the diagonal...
    # Isn't there a conjugate missing in the original formulation?
    else:
        D = D + np.sum(X1 * X1.conj(), axis=0)[..., None]
        D = D + np.sum(X2 * X2.conj(), axis=0)
    return D
