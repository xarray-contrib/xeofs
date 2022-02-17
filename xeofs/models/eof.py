from typing import Optional, Union, Iterable

import numpy as np

from xeofs.models._eof_base import _EOF_base
from xeofs.models._array_transformer import _ArrayTransformer

import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


class EOF(_EOF_base):
    '''EOF analysis of a single ``np.ndarray``.

    Parameters
    ----------
    X : np.ndarray
        Data to be decpomposed. ``X`` can be any N-dimensional array with the
        first dimension containing the variable whose variance is to be
        maximised. All remaining dimensions will be automatically reshaped to
        obtain a 2D matrix.
    n_modes : Optional[int]
        Number of modes to compute. Computing less modes can results in
        performance gains. If None, then the maximum number of modes is
        equivalent to ``min(n_samples, n_features)`` (the default is None).
    norm : bool
        Normalize each feature (e.g. grid cell) by its temporal standard
        deviation (the default is False).
    axis : Union[int, Iterable[int]]
        Axis along which variance should be maximised. Can also be
        multi-dimensional. For example, given a data array of dimensions
        ``(n x p1 x p2)`` with `n` time series at ``p1`` and ``p2`` different
        locations, ``axis=0`` will maximise `temporal` variance along ``n``.
        In contrast, ``axis=[1, 2]`` will maximise `spatial` variance along
        ``(p1 x p2)`` (the default is 0).


    Examples
    --------

    Import package and create data:

    >>> import numpy as np
    >>> from xeofs.models import EOF
    >>> rng = np.random.default_rng(7)
    >>> X = rng.standard_normal((4, 2, 3))

    Initialize standardized EOF analysis and compute the first 2 modes:

    >>> model = EOF(X, norm=True, n_modes=2)
    >>> model.solve()

    Get explained variance:

    >>> model.explained_variance()
    ... array([4.81848833, 2.20765019])

    Get EOFs:

    >>> model.eofs()
    ... array([[[ 0.52002146, -0.00698575],
    ...         [ 0.41059796, -0.31603563]],
    ...
    ...         [[ 0.32091783,  0.47647144],
    ...         [-0.51771611,  0.01380736]],
    ...
    ...         [[-0.28840959,  0.63341346],
    ...         [ 0.32678537,  0.52119516]]])

    Get PCs:

    >>> model.pcs()
    ... array([[ 1.76084189, -0.92030205],
    ...        [-2.13581896, -1.49704775],
    ...        [ 2.02091078,  0.65502667],
    ...        [-1.64593371,  1.76232312]])

    '''

    def __init__(
        self,
        X: np.ndarray,
        n_modes : Optional[int] = None,
        norm : bool = False,
        axis : Union[int, Iterable[int]] = 0
    ):

        self._tf = _ArrayTransformer()
        X = self._tf.fit_transform(X, axis=axis)

        super().__init__(
            X=X,
            n_modes=n_modes,
            norm=norm
        )

    def eofs(self):
        eofs = super().eofs()
        return self._tf.back_transform_eofs(eofs)

    def pcs(self):
        pcs = super().pcs()
        return self._tf.back_transform_pcs(pcs)
