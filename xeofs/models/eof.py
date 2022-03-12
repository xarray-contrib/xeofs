from typing import Optional, Union, Iterable, Tuple, List

import numpy as np

from xeofs.models._base_eof import _BaseEOF
from xeofs.models._transformer import _MultiArrayTransformer
from ..utils.tools import squeeze

import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

Array = np.ndarray
ArrayList = Union[Array, List[Array]]


class EOF(_BaseEOF):
    '''EOF analysis of a single ``np.ndarray``.

    Parameters
    ----------
    X : np.ndarray
        Data matrix to be decomposed. ``X`` can be any N-dimensional array.
        Dimensions whose variance shall be maximised (denoted as *samples*)
        have to be defined by the ``axis`` parameter. All remaining axes will
        be reshaped into a new axis called *features*.
    n_modes : Optional[int]
        Number of modes to compute. Computing less modes can results in
        performance gains. If None, then the maximum number of modes is
        equivalent to ``min(n_samples, n_features)`` (the default is None).
    norm : bool
        Normalize each feature (e.g. grid cell) by its temporal standard
        deviation (the default is False).
    weights : Optional[np.ndarray] = None
        Weights applied to features. Must have the same dimensions as the
        original features which are the remaining axes not specified by
        ``axis`` parameter).
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
        X: ArrayList,
        axis : Union[int, Iterable[int]] = 0,
        n_modes : Optional[int] = None,
        norm : bool = False,
        weights : Optional[ArrayList] = None,
    ):

        self._tf = _MultiArrayTransformer()
        X = self._tf.fit_transform(X)
        weights = self._tf.transform_weights(weights)

        super().__init__(
            X=X,
            n_modes=n_modes,
            norm=norm,
            weights=weights
        )

    def eofs(self, scaling : int = 0) -> ArrayList:
        eofs = super().eofs(scaling=scaling)
        eofs = self._tf.back_transform_eofs(eofs)
        return squeeze(eofs)

    def pcs(self, scaling : int = 0) -> Array:
        pcs = super().pcs(scaling=scaling)
        return self._tf.back_transform_pcs(pcs)

    def eofs_as_correlation(self) -> Tuple[ArrayList, ArrayList]:
        corr, pvals = super().eofs_as_correlation()
        corr = self._tf.back_transform_eofs(corr)
        pvals = self._tf.back_transform_eofs(pvals)
        return squeeze(corr), squeeze(pvals)

    def reconstruct_X(
        self,
        mode : Optional[Union[int, List[int], slice]] = None
    ) -> ArrayList:
        Xrec = super().reconstruct_X(mode)
        Xrec = self._tf.back_transform(Xrec)
        return squeeze(Xrec)

    def project_onto_eofs(
        self,
        X : ArrayList,
        scaling : int = 0
    ) -> Array:
        '''Project new data onto the EOFs.

        Parameters
        ----------
        X : array or list of arrays
             New data to project. Data must have same feature shape as original
             data.
        scaling : [0, 1, 2]
            Projections are scaled (i) to be orthonormal (``scaling=0``), (ii) by the
            square root of the eigenvalues (``scaling=1``) or (iii) by the
            singular values (``scaling=2``). In case no weights were applied,
            scaling by the singular values results in the projections having the
            unit of the input data (the default is 0).

        '''
        proj = _MultiArrayTransformer()
        X_proj = proj.fit_transform(X, axis=self._tf.axis_samples)
        pcs = super().project_onto_eofs(X=X_proj, scaling=scaling)
        return proj.back_transform_pcs(pcs)
