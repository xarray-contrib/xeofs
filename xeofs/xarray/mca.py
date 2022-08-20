from typing import Optional, Tuple, Union, List

import pandas as pd
import xarray as xr

from ..models._base_mca import _BaseMCA
from ..utils.tools import squeeze
from ._transformer import _MultiDataArrayTransformer

DataArray = xr.DataArray
DataArrayList = Union[DataArray, List[DataArray]]


class MCA(_BaseMCA):
    '''Maximum Covariance Analysis of two or multiple ``xr.DataArray``.

    Parameters
    ----------
    X, Y : DataArrayList
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
    weights_X, weights_Y : Optional[DataArrayList] = None
        Weights applied to features of ``X`` and ``Y``. Must have the same
        dimensions as the original features which are the remaining axes
        not specified by ``axis`` parameter).


    Examples
    --------

    Import package and create data:

    >>> import xarray as xr
    >>> from xeofs.xarray import MCA

    Initialize standardized MCA and compute the first 2 modes of two
    ``xr.DataArray`` ``da1`` and ``da2``:

    >>> model = MCA(da1, da2, norm=True, n_modes=2)
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
        X: DataArrayList,
        Y: DataArrayList,
        dim : Optional[Union[str, List[str]]] = 'time',
        n_modes : Optional[int] = None,
        norm : bool = False,
        weights_X : Optional[Union[str, DataArrayList]] = None,
        weights_Y : Optional[Union[str, DataArrayList]] = None
    ):
        use_coslat_X = True if weights_X == 'coslat' else False
        use_coslat_Y = True if weights_Y == 'coslat' else False

        self._tfx = _MultiDataArrayTransformer()
        self._tfy = _MultiDataArrayTransformer()

        X = self._tfx.fit_transform(X, dim=dim, coslat=use_coslat_X)
        Y = self._tfy.fit_transform(Y, dim=dim, coslat=use_coslat_Y)

        if use_coslat_X:
            weights_X = self._tfx.coslat_weights
        if use_coslat_Y:
            weights_Y = self._tfy.coslat_weights

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

    def singular_values(self) -> DataArray:
        svalues = super().singular_values()
        svalues = xr.DataArray(
            svalues,
            dims=['mode'],
            coords=dict(mode=self._idx_mode),
            name='singular_values'
        )
        return svalues

    def explained_covariance(self) -> DataArray:
        expvar = super().explained_covariance()
        expvar = xr.DataArray(
            expvar,
            dims=['mode'],
            coords=dict(mode=self._idx_mode),
            name='explained_covariance'
        )
        return expvar

    def squared_covariance_fraction(self) -> DataArray:
        scf = super().squared_covariance_fraction()
        scf = xr.DataArray(
            scf,
            dims=['mode'],
            coords=dict(mode=self._idx_mode),
            name='squared_covariance_fraction'
        )
        return scf

    def singular_vectors(
            self, scaling : int = 0
    ) -> Tuple[DataArrayList, DataArrayList]:
        Vx, Vy = super().singular_vectors(scaling=scaling)
        Vx = self._tfx.back_transform_eofs(Vx)
        Vy = self._tfy.back_transform_eofs(Vy)
        # Rename DataArrays
        for da in Vx:
            da.name = 'left_singular_vectors'
        for da in Vy:
            da.name = 'right_singular_vectors'
        return squeeze(Vx), squeeze(Vy)

    def pcs(self, scaling : int = 0) -> Tuple[DataArray, DataArray]:
        Ux, Uy = super().pcs(scaling=scaling)
        Ux = self._tfx.back_transform_pcs(Ux)
        Uy = self._tfy.back_transform_pcs(Uy)
        # Rename DataArrays
        Ux.name = 'left_pcs'
        Uy.name = 'right_pcs'
        return Ux, Uy

    def homogeneous_patterns(self) -> Tuple[DataArrayList, DataArrayList]:
        hom_pats, pvals = super().homogeneous_patterns()
        hom_patsx = self._tfx.back_transform_eofs(hom_pats[0])
        pvalsx = self._tfx.back_transform_eofs(pvals[0])
        hom_patsy = self._tfy.back_transform_eofs(hom_pats[1])
        pvalsy = self._tfy.back_transform_eofs(pvals[1])

        # Rename DataArrays
        for da, p in zip(hom_patsx, pvalsx):
            da.name = 'left_homogeneous_patterns'
            p.name = 'left_homogeneous_patterns_p_values'
        for da, p in zip(hom_patsy, pvalsy):
            da.name = 'right_homogeneous_patterns'
            p.name = 'right_homogeneous_patterns_p_values'

        # Remove list if single DataArray
        hom_patsx = squeeze(hom_patsx)
        hom_patsy = squeeze(hom_patsy)
        pvalsx = squeeze(pvalsx)
        pvalsy = squeeze(pvalsy)
        return (hom_patsx, hom_patsy), (pvalsx, pvalsy)

    def heterogeneous_patterns(self) -> Tuple[DataArrayList, DataArrayList]:
        het_pats, pvals = super().heterogeneous_patterns()
        het_patsx = self._tfx.back_transform_eofs(het_pats[0])
        pvalsx = self._tfx.back_transform_eofs(pvals[0])
        het_patsy = self._tfy.back_transform_eofs(het_pats[1])
        pvalsy = self._tfy.back_transform_eofs(pvals[1])

        # Rename DataArrays
        for da, p in zip(het_patsx, pvalsx):
            da.name = 'left_heterogeneous_patterns'
            p.name = 'left_heterogeneous_patterns_p_values'
        for da, p in zip(het_patsy, pvalsy):
            da.name = 'right_heterogeneous_patterns'
            p.name = 'right_heterogeneous_patterns_p_values'

        # Remove list if single DataArray
        het_patsx = squeeze(het_patsx)
        het_patsy = squeeze(het_patsy)
        pvalsx = squeeze(pvalsx)
        pvalsy = squeeze(pvalsy)
        return (het_patsx, het_patsy), (pvalsx, pvalsy)

    def reconstruct_XY(
        self,
        mode : Optional[Union[int, List[int], slice]] = None
    ) -> Tuple[DataArrayList, DataArrayList]:
        Xrec, Yrec = super().reconstruct_XY(mode)
        Xrec = self._tfx.back_transform(Xrec)
        Yrec = self._tfy.back_transform(Yrec)
        return squeeze(Xrec), squeeze(Yrec)

    def project_onto_left_singular_vectors(
        self,
        X : DataArrayList = None,
        scaling : int = 0
    ) -> DataArray:
        # Transform data to 2D
        projx = _MultiDataArrayTransformer()
        X_proj = projx.fit_transform(X, dim=self._tfy.dims_samples)
        # Perform projection
        pcs_X = super().project_onto_left_singular_vectors(
            X=X_proj, scaling=scaling
        )
        # Backtransform to PC format
        pcs_X = projx.back_transform_pcs(pcs_X)
        return pcs_X

    def project_onto_right_singular_vectors(
        self,
        Y : DataArrayList = None,
        scaling : int = 0
    ) -> DataArray:
        # Transform data to 2D
        projy = _MultiDataArrayTransformer()
        Y_proj = projy.fit_transform(Y, dim=self._tfy.dims_samples)
        # Perform projection
        pcs_Y = super().project_onto_right_singular_vectors(
            Y=Y_proj, scaling=scaling
        )
        # Backtransform to PC format
        pcs_Y = projy.back_transform_pcs(pcs_Y)
        return pcs_Y
