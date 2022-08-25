from typing import Optional, Union, Iterable, Tuple, List

import numpy as np

from xeofs.models._base_mca import _BaseMCA
from xeofs.models._transformer import _MultiArrayTransformer
from ..utils.tools import squeeze

import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

Array = np.ndarray
ArrayList = Union[Array, List[Array]]


class MCA(_BaseMCA):
    '''Maximum Covariance Analysis (MCA) of a single or multiple ``np.ndarray``.

    MCA is also known as Singular Value Decomposition (SVD) analysis or
    Partial Least Squares (PLS) analysis.

    Parameters
    ----------
    X, Y : np.ndarray
        Data matrices two compute maximum covariance. ``X`` and ``Y`` can be
        any N-dimensional array. Dimensions along which covariance shall be
        maximised (denoted as *samples*) have to be defined by the
        ``axis`` parameter. All remaining axes will be reshaped into a new
        axis called *features*. Sample dimension of ``X`` and ``Y`` must be
        the same.
    n_modes : Optional[int]
        Number of modes to compute. Computing less modes can results in
        performance gains. If None, then the maximum number of modes is
        equivalent to ``min(n_samples, n_features)`` (the default is None).
    norm : bool
        Normalize each feature (e.g. grid cell) by its temporal standard
        deviation (the default is False).
    weights_X : Optional[np.ndarray] = None
        Weights applied to features of ``X``. Must have the same dimensions as the
        original features which are the remaining axes not specified by
        ``axis`` parameter).
    weights_Y : Optional[np.ndarray] = None
        Weights applied to features of ``Y``. Must have the same dimensions as the
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
    >>> from xeofs.models import MCA
    >>> rng = np.random.default_rng(7)
    >>> X = rng.standard_normal((45, 4, 3))
    >>> Y = rng.standard_normal((45, 2, 5))

    Initialize standardized MCA analysis and compute the first 2 modes:

    >>> model = MCA(X, norm=True, n_modes=2, axis=0)
    >>> model.solve()

    Get singular vectors:

    >>> model.singular_vectors()

    Get homogeneous patterns:

    >>> model.homogeneous_patterns()

    '''
    def __init__(
        self,
        X: ArrayList,
        Y: ArrayList,
        axis : Union[int, Iterable[int]] = 0,
        n_modes : Optional[int] = None,
        norm : bool = False,
        weights_X : Optional[ArrayList] = None,
        weights_Y : Optional[ArrayList] = None,
    ):

        self._tfx = _MultiArrayTransformer()
        self._tfy = _MultiArrayTransformer()
        X = self._tfx.fit_transform(X, axis=axis)
        Y = self._tfy.fit_transform(Y, axis=axis)
        weights_X = self._tfx.transform_weights(weights_X)
        weights_Y = self._tfy.transform_weights(weights_Y)

        super().__init__(
            X=X,
            Y=Y,
            n_modes=n_modes,
            norm=norm,
            weights_X=weights_X,
            weights_Y=weights_Y
        )

    def singular_values(self) -> Array:
        return super().singular_values()

    def explained_covariance(self) -> Array:
        return super().explained_covariance()

    def squared_covariance_fraction(self) -> Array:
        return super().squared_covariance_fraction()

    def singular_vectors(
            self, scaling : int = 0
    ) -> Tuple[ArrayList, ArrayList]:
        Vx, Vy = super().singular_vectors(scaling=scaling)
        Vx = self._tfx.back_transform_eofs(Vx)
        Vy = self._tfy.back_transform_eofs(Vy)
        return squeeze(Vx), squeeze(Vy)

    def pcs(self, scaling : int = 0) -> Tuple[ArrayList, ArrayList]:
        Ux, Uy = super().pcs(scaling=scaling)
        Ux = self._tfx.back_transform_pcs(Ux)
        Uy = self._tfy.back_transform_pcs(Uy)
        return Ux, Uy

    def homogeneous_patterns(self) -> Tuple[ArrayList, ArrayList]:
        hom_pats, pvals = super().homogeneous_patterns()
        hom_patsx = squeeze(self._tfx.back_transform_eofs(hom_pats[0]))
        hom_patsy = squeeze(self._tfy.back_transform_eofs(hom_pats[1]))
        pvalsx = squeeze(self._tfx.back_transform_eofs(pvals[0]))
        pvalsy = squeeze(self._tfy.back_transform_eofs(pvals[1]))
        return (hom_patsx, hom_patsy), (pvalsx, pvalsy)

    def heterogeneous_patterns(self) -> Tuple[ArrayList, ArrayList]:
        het_pats, pvals = super().heterogeneous_patterns()
        het_patsx = squeeze(self._tfx.back_transform_eofs(het_pats[0]))
        het_patsy = squeeze(self._tfy.back_transform_eofs(het_pats[1]))
        pvalsx = squeeze(self._tfx.back_transform_eofs(pvals[0]))
        pvalsy = squeeze(self._tfy.back_transform_eofs(pvals[1]))
        return (het_patsx, het_patsy), (pvalsx, pvalsy)

    def reconstruct_XY(
        self,
        mode : Optional[Union[int, List[int], slice]] = None
    ) -> Tuple[ArrayList, ArrayList]:
        Xrec, Yrec = super().reconstruct_XY(mode)
        Xrec = self._tfx.back_transform(Xrec)
        Yrec = self._tfy.back_transform(Yrec)
        return squeeze(Xrec), squeeze(Yrec)

    def project_onto_left_singular_vectors(
        self,
        X : ArrayList = None,
        scaling : int = 0
    ) -> Array:
        # Transform data to 2D
        projx = _MultiArrayTransformer()
        X_proj = projx.fit_transform(X, axis=self._tfx.axis_samples)
        # Perform projection
        pcs_X = super().project_onto_left_singular_vectors(
            X=X_proj, scaling=scaling
        )
        # Backtransform to PC format
        pcs_X = projx.back_transform_pcs(pcs_X)
        return pcs_X

    def project_onto_right_singular_vectors(
        self,
        Y : ArrayList = None,
        scaling : int = 0
    ) -> Array:
        # Transform data to 2D
        projy = _MultiArrayTransformer()
        Y_proj = projy.fit_transform(Y, axis=self._tfy.axis_samples)
        # Perform projection
        pcs_Y = super().project_onto_right_singular_vectors(
            Y=Y_proj, scaling=scaling
        )
        # Backtransform to PC format
        pcs_Y = projy.back_transform_pcs(pcs_Y)
        return pcs_Y
