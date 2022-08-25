from typing import Optional, Tuple, List, Union, Iterable

import xarray as xr

from ..models._base_rock_pca import _BaseROCK_PCA
from ..utils.tools import squeeze
from ._transformer import _MultiDataArrayTransformer

DataArray = xr.DataArray
DataArrayList = Union[DataArray, List[DataArray]]


class ROCK_PCA(_BaseROCK_PCA):
    '''ROCK-PCA of a single or multiple ``xr.DataArray``.'''

    def __init__(
        self,
        X: DataArray,
        n_rot : int,
        power : int,
        sigma : float,
        dim: Union[str, Iterable[str]] = 'time',
        n_modes: Optional[int] = None,
        norm: bool = False,
        weights: Optional[DataArray] = None,
    ):
        use_coslat = True if weights == 'coslat' else False

        self._tf = _MultiDataArrayTransformer()
        X = self._tf.fit_transform(X, dim=dim, coslat=use_coslat)

        if use_coslat:
            weights = self._tf.coslat_weights
        weights = self._tf.transform_weights(weights)

        super().__init__(
            X=X, n_rot=n_rot, power=power, sigma=sigma, n_modes=n_modes,
            norm=norm, weights=weights
        )
        self._idx_mode = xr.IndexVariable('mode', range(1, self._params['n_modes'] + 1))
        self._dim = dim

    def explained_variance(self) -> DataArray:
        expvar = super().explained_variance()
        return xr.DataArray(
            expvar,
            dims=['mode'],
            coords={'mode' : self._idx_mode},
            name='explained_variance'
        )

    def explained_variance_ratio(self) -> DataArray:
        expvar = super().explained_variance_ratio()
        return xr.DataArray(
            expvar,
            dims=['mode'],
            coords={'mode' : self._idx_mode},
            name='explained_variance_ratio'
        )

    def eofs(self) -> DataArrayList:
        eofs = super().eofs()
        eofs = self._tf.back_transform_eofs(eofs)
        return squeeze(eofs)

    def pcs(self) -> DataArray:
        pcs = super().pcs()
        return self._tf.back_transform_pcs(pcs)

    def eofs_amplitude(self) -> DataArrayList:
        amp = super().eofs_amplitude()
        amp = self._tf.back_transform_eofs(amp)
        for da in amp:
            da.name = 'eofs_amplitude'
        return squeeze(amp)

    def pcs_amplitude(self) -> DataArray:
        amp = super().pcs_amplitude()
        amp = self._tf.back_transform_pcs(amp)
        amp.name = 'pcs_amplitude'
        return amp

    def eofs_phase(self) -> DataArrayList:
        phase = super().eofs_phase()
        phase = self._tf.back_transform_eofs(phase)
        for da in phase:
            da.name = 'eofs_phase'
        return squeeze(phase)

    def pcs_phase(self) -> DataArray:
        phase = super().pcs_phase()
        phase = self._tf.back_transform_pcs(phase)
        phase.name = 'pcs_phase'
        return phase
