from typing import Optional, Union, Iterable, Tuple, List

import numpy as np

from xeofs.models._base_eof import _BaseEOF
from xeofs.models._array_transformer import _ArrayTransformer

import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


class MultivariateEOF(_BaseEOF):
    '''EOF analysis of a multiple ``np.ndarray``.

    Parameters
    ----------
    X : List[np.ndarray]
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
    weights : Optional[List[np.ndarray]] = None
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



    '''

    def __init__(
        self,
        X: List[np.ndarray],
        axis : Union[int, Iterable[int]] = 0,
        n_modes : Optional[int] = None,
        norm : bool = False,
        weights : Optional[List[np.ndarray]] = None,
    ):
        self._tf = []
        X_transformed = []
        weights_transformed = []
        for x in X:
            tf = _ArrayTransformer()
            X_transformed.append(tf.fit_transform(x, axis=axis))
            if weights is not None:
                weights_transformed.append(tf.transform_weights(weights))
            self._tf.append(tf)

        # TODO: raise error if different sample length

        shapes = [x.shape[1] for x in X_transformed]
        self._multi_idx_features = np.insert(np.cumsum(shapes), 0, 0)
        X = np.concatenate(X_transformed, axis=1)
        weights = np.concatenate(weights_transformed, axis=1) if weights is not None else weights

        super().__init__(
            X=X,
            n_modes=n_modes,
            norm=norm,
            weights=weights
        )

    def eofs(self, scaling : int = 0) -> np.ndarray:
        transformers = self._tf
        idx = self._multi_idx_features

        eofs = super().eofs(scaling=scaling)
        eofs = [eofs[idx[i]:idx[i + 1]] for i in range(len(idx) - 1)]
        return [tf.back_transform_eofs(eof) for eof, tf in zip(eofs, transformers)]

    def pcs(self, scaling : int = 0) -> np.ndarray:
        pcs = super().pcs(scaling=scaling)
        return self._tf[0].back_transform_pcs(pcs)

    def eofs_as_correlation(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()
        corr, pvals = super().eofs_as_correlation()
        corr = self._tf.back_transform_eofs(corr)
        pvals = self._tf.back_transform_eofs(pvals)
        return corr, pvals

    def reconstruct_X(
        self,
        mode : Optional[Union[int, List[int], slice]] = None
    ) -> np.ndarray:
        raise NotImplementedError()
        Xrec = super().reconstruct_X(mode)
        return self._tf.back_transform(Xrec)

    def project_onto_eofs(
        self,
        X : np.ndarray,
        scaling : int = 0
    ) -> np.ndarray:
        '''Project new data onto the EOFs.

        Parameters
        ----------
        X : np.ndarray
             New data to project. Data must have same feature shape as original
             data.
        scaling : [0, 1, 2]
            Projections are scaled (i) to be orthonormal (``scaling=0``), (ii) by the
            square root of the eigenvalues (``scaling=1``) or (iii) by the
            singular values (``scaling=2``). In case no weights were applied,
            scaling by the singular values results in the projections having the
            unit of the input data (the default is 0).

        '''
        raise NotImplementedError()
        proj = _ArrayTransformer()
        X = proj.fit_transform(X, axis=self._tf.axis_samples)
        pcs = super().project_onto_eofs(X=X, scaling=scaling)
        return proj.back_transform_pcs(pcs)
