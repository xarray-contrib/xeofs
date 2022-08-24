import xarray as xr
from typing import Optional, Union, List, Tuple

from ._transformer import _MultiDataArrayTransformer
from ..models._base_mca_rotator import _BaseMCARotator
from ..utils.tools import squeeze

DataArray = xr.DataArray
DataArrayList = Union[DataArray, List[DataArray]]


class MCA_Rotator(_BaseMCARotator):
    '''Rotates a solution obtained from ``xe.xarray.MCA``.'''

    def __init__(
        self,
        n_rot : int,
        loadings : str = 'standard',
        power : int = 1,
        max_iter : int = 1000,
        rtol : float = 1e-8
    ):

        super().__init__(
            n_rot=n_rot, loadings=loadings, power=power,
            max_iter=max_iter, rtol=rtol
        )

    def singular_values(self) -> DataArray:
        n_rot = self._params['n_rot']
        return xr.DataArray(
            super().singular_values(),
            dims=['mode'],
            coords=dict(mode=self._model._idx_mode[:n_rot]),
            name='singular_values'
        )

    def explained_covariance(self) -> DataArray:
        n_rot = self._params['n_rot']
        return xr.DataArray(
            super().explained_covariance(),
            dims=['mode'],
            coords=dict(mode=self._model._idx_mode[:n_rot]),
            name='explained_covariance'
        )

    def squared_covariance_fraction(self) -> DataArray:
        n_rot = self._params['n_rot']
        return xr.DataArray(
            super().squared_covariance_fraction(),
            dims=['mode'],
            coords=dict(mode=self._model._idx_mode[:n_rot]),
            name='squared_covariance_fraction'
        )

    def singular_vectors(self, scaling : int = 0) -> DataArrayList:
        Vx, Vy = super().singular_vectors(scaling=scaling)
        Vx = self._model._tfx.back_transform_eofs(Vx)
        Vy = self._model._tfy.back_transform_eofs(Vy)
        for x in Vx:
            x.name = 'left_singular_vectors'
        for y in Vy:
            y.name = 'right_singular_vectors'
        return squeeze(Vx), squeeze(Vy)

    def pcs(self, scaling : int = 0) -> DataArray:
        Ux, Uy = super().pcs(scaling=scaling)
        Ux = self._model._tfx.back_transform_pcs(Ux)
        Uy = self._model._tfy.back_transform_pcs(Uy)
        Ux.name = 'left_pcs'
        Uy.name = 'right_pcs'
        return Ux, Uy

    def homogeneous_patterns(self) -> Tuple[DataArrayList, DataArrayList]:
        hom_pats, pvals = super().homogeneous_patterns()
        hom_pats_x = self._model._tfx.back_transform_eofs(hom_pats[0])
        hom_pats_y = self._model._tfy.back_transform_eofs(hom_pats[1])
        pvals_x = self._model._tfx.back_transform_eofs(pvals[0])
        pvals_y = self._model._tfy.back_transform_eofs(pvals[1])

        for x, p in zip(hom_pats_x, pvals_x):
            x.name = 'left_homogeneous_patterns'
            p.name = 'left_homogeneous_patterns_p_values'
        for x, p in zip(hom_pats_y, pvals_y):
            x.name = 'right_homogeneous_patterns'
            p.name = 'right_homogeneous_patterns_p_values'

        hom_pats_x = squeeze(hom_pats_x)
        hom_pats_y = squeeze(hom_pats_y)
        pvals_x = squeeze(pvals_x)
        pvals_y = squeeze(pvals_y)
        return (hom_pats_x, hom_pats_y), (pvals_x, pvals_y)

    def heterogeneous_patterns(self) -> Tuple[DataArrayList, DataArrayList]:
        het_pats, pvals = super().heterogeneous_patterns()
        het_pats_x = self._model._tfx.back_transform_eofs(het_pats[0])
        het_pats_y = self._model._tfy.back_transform_eofs(het_pats[1])
        pvals_x = self._model._tfx.back_transform_eofs(pvals[0])
        pvals_y = self._model._tfy.back_transform_eofs(pvals[1])

        for x, p in zip(het_pats_x, pvals_x):
            x.name = 'left_heterogeneous_patterns'
            p.name = 'left_heterogeneous_patterns_p_values'
        for x, p in zip(het_pats_y, pvals_y):
            x.name = 'right_heterogeneous_patterns'
            p.name = 'right_heterogeneous_patterns_p_values'

        het_pats_x = squeeze(het_pats_x)
        het_pats_y = squeeze(het_pats_y)
        pvals_x = squeeze(pvals_x)
        pvals_y = squeeze(pvals_y)
        return (het_pats_x, het_pats_y), (pvals_x, pvals_y)

    def reconstruct_XY(
        self,
        mode : Optional[Union[int, List[int], slice]] = None
    ) -> DataArrayList:
        Xrec, Yrec = super().reconstruct_XY(mode=mode)
        Xrec = self._model._tfx.back_transform(Xrec)
        Yrec = self._model._tfy.back_transform(Yrec)
        return squeeze(Xrec), squeeze(Yrec)

    def project_onto_left_singular_vectors(
        self,
        X : DataArrayList,
        scaling : int = 0
    ) -> DataArray:
        proj = _MultiDataArrayTransformer()
        X = proj.fit_transform(X, dim=self._model._tfx.dims_samples)
        pcs = super().project_onto_left_singular_vectors(X=X, scaling=scaling)
        return proj.back_transform_pcs(pcs)

    def project_onto_right_singular_vectors(
        self,
        Y : DataArrayList,
        scaling : int = 0
    ) -> DataArray:
        proj = _MultiDataArrayTransformer()
        Y = proj.fit_transform(Y, dim=self._model._tfy.dims_samples)
        pcs = super().project_onto_right_singular_vectors(Y=Y, scaling=scaling)
        return proj.back_transform_pcs(pcs)
