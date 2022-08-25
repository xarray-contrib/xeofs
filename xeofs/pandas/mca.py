from typing import Optional, Tuple, Union, List

import pandas as pd

from ..models._base_mca import _BaseMCA
from ..utils.tools import squeeze
from ._transformer import _MultiDataFrameTransformer

DataFrame = pd.DataFrame
DataFrameList = Union[DataFrame, List[DataFrame]]


class MCA(_BaseMCA):
    '''Maximum Covariance Analysis of two or multiple ``pd.DataFrame``.

    Parameters
    ----------
    X, Y : DataFrameList
        Data for which to maximise covariance.
    axis : int
        Axis along which variance should be maximsed (the default is 0).
    n_modes : Optional[int]
        Number of modes to compute. Computing less modes can results in
        performance gains. If None, then the maximum number of modes is
        equivalent to ``min(n_samples, n_features)`` (the default is None).
    norm : bool
        Normalize each feature (e.g. grid cell) by its temporal standard
        deviation (the default is False).
    weights_X, weights_Y : Optional[DataFrameList] = None
        Weights applied to features of ``X`` and ``Y``. Must have the same
        dimensions as the original features which are the remaining axes
        not specified by ``axis`` parameter).


    Examples
    --------

    Import package and create data:

    >>> import pandas as pd
    >>> from xeofs.pandas import MCA
    >>> rng = np.random.default_rng(7)
    >>> X = rng.standard_normal((14, 3))
    >>> Y = rng.standard_normal((14, 4))
    >>> dfX = pd.DataFrame(X)
    >>> dfY = pd.DataFrame(Y)

    Initialize standardized MCA and compute the first 2 modes:

    >>> model = MCA(dfX, dfY, norm=True, n_modes=2)
    >>> model.solve()

    Get squared covariance fraction:

    >>> model.squared_covariance_fraction()

    Get singular vectors:

    >>> left_vectors, right_vectors = model.singular_vectors()

    Get PCs:

    >>> left_pcs, right_pcs = model.pcs()

    '''

    def __init__(
        self,
        X: DataFrameList,
        Y: DataFrameList,
        axis : int = 0,
        n_modes : Optional[int] = None,
        norm : bool = False,
        weights_X : Optional[DataFrameList] = None,
        weights_Y : Optional[DataFrameList] = None
    ):

        self._tfx = _MultiDataFrameTransformer()
        self._tfy = _MultiDataFrameTransformer()
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
        self._idx_mode = pd.Index(range(1, self.n_modes + 1), name='mode')

    def singular_values(self) -> DataFrame:
        svalues = super().singular_values()
        svalues = pd.Series(
            svalues,
            index=self._idx_mode,
            name='singular_values',
        )
        return svalues

    def explained_covariance(self) -> DataFrame:
        expvar = super().explained_covariance()
        expvar = pd.Series(
            expvar,
            index=self._idx_mode,
            name='explained_covariance',
        )
        return expvar

    def squared_covariance_fraction(self) -> DataFrame:
        scf = super().squared_covariance_fraction()
        scf = pd.Series(
            scf,
            index=self._idx_mode,
            name='squared_covariance_fraction',
        )
        return scf

    def singular_vectors(
            self, scaling : int = 0
    ) -> Tuple[DataFrameList, DataFrameList]:
        Vx, Vy = super().singular_vectors(scaling=scaling)
        Vx = self._tfx.back_transform_eofs(Vx)
        Vy = self._tfy.back_transform_eofs(Vy)
        return squeeze(Vx), squeeze(Vy)

    def pcs(self, scaling : int = 0) -> Tuple[DataFrame, DataFrame]:
        Ux, Uy = super().pcs(scaling=scaling)
        Ux = self._tfx.back_transform_pcs(Ux)
        Uy = self._tfy.back_transform_pcs(Uy)
        return Ux, Uy

    def homogeneous_patterns(self) -> Tuple[DataFrameList, DataFrameList]:
        hom_pats, pvals = super().homogeneous_patterns()
        hom_patsx = squeeze(self._tfx.back_transform_eofs(hom_pats[0]))
        hom_patsy = squeeze(self._tfy.back_transform_eofs(hom_pats[1]))
        pvalsx = squeeze(self._tfx.back_transform_eofs(pvals[0]))
        pvalsy = squeeze(self._tfy.back_transform_eofs(pvals[1]))
        return (hom_patsx, hom_patsy), (pvalsx, pvalsy)

    def heterogeneous_patterns(self) -> Tuple[DataFrameList, DataFrameList]:
        het_pats, pvals = super().heterogeneous_patterns()
        het_patsx = squeeze(self._tfx.back_transform_eofs(het_pats[0]))
        het_patsy = squeeze(self._tfy.back_transform_eofs(het_pats[1]))
        pvalsx = squeeze(self._tfx.back_transform_eofs(pvals[0]))
        pvalsy = squeeze(self._tfy.back_transform_eofs(pvals[1]))
        return (het_patsx, het_patsy), (pvalsx, pvalsy)

    def reconstruct_XY(
        self,
        mode : Optional[Union[int, List[int], slice]] = None
    ) -> Tuple[DataFrameList, DataFrameList]:
        Xrec, Yrec = super().reconstruct_XY(mode)
        Xrec = self._tfx.back_transform(Xrec)
        Yrec = self._tfy.back_transform(Yrec)
        return squeeze(Xrec), squeeze(Yrec)

    def project_onto_left_singular_vectors(
        self,
        X : DataFrameList = None,
        scaling : int = 0
    ) -> DataFrame:
        # Transform data to 2D
        projx = _MultiDataFrameTransformer()
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
        Y : DataFrameList = None,
        scaling : int = 0
    ) -> DataFrame:
        # Transform data to 2D
        projy = _MultiDataFrameTransformer()
        Y_proj = projy.fit_transform(Y, axis=self._tfy.axis_samples)
        # Perform projection
        pcs_Y = super().project_onto_right_singular_vectors(
            Y=Y_proj, scaling=scaling
        )
        # Backtransform to PC format
        pcs_Y = projy.back_transform_pcs(pcs_Y)
        return pcs_Y
