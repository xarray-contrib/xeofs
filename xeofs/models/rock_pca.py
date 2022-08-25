from typing import Optional, Tuple, List, Union, Iterable

import numpy as np

from ._base_rock_pca import _BaseROCK_PCA
from xeofs.models._transformer import _MultiArrayTransformer
from ..utils.tools import squeeze

Array = np.ndarray
ArrayList = Union[Array, List[Array]]


class ROCK_PCA(_BaseROCK_PCA):
    '''ROCK-PCA of a single or multiple ``np.ndarray``.'''

    def __init__(
        self,
        X: np.ndarray,
        n_rot : int,
        power : int,
        sigma : float,
        axis : Union[int, Iterable[int]] = 0,
        n_modes: Optional[int] = None,
        norm: bool = False,
        weights: Optional[np.ndarray] = None,
    ):
        self._tf = _MultiArrayTransformer()
        X = self._tf.fit_transform(X, axis=axis)
        weights = self._tf.transform_weights(weights)

        super().__init__(
            X=X, n_rot=n_rot, power=power, sigma=sigma, n_modes=n_modes,
            norm=norm, weights=weights
        )

    def explained_variance(self) -> Array:
        return super().explained_variance()

    def explained_variance_ratio(self) -> Array:
        return super().explained_variance_ratio()

    def eofs(self) -> ArrayList:
        eofs = super().eofs()
        eofs = self._tf.back_transform_eofs(eofs)
        return squeeze(eofs)

    def pcs(self) -> Array:
        pcs = super().pcs()
        return self._tf.back_transform_pcs(pcs)

    def eofs_amplitude(self) -> ArrayList:
        amp = super().eofs_amplitude()
        amp = self._tf.back_transform_eofs(amp)
        return squeeze(amp)

    def pcs_amplitude(self) -> Array:
        amp = super().pcs_amplitude()
        return self._tf.back_transform_pcs(amp)

    def eofs_phase(self) -> ArrayList:
        phase = super().eofs_phase()
        phase = self._tf.back_transform_eofs(phase)
        return squeeze(phase)

    def pcs_phase(self) -> Array:
        phase = super().pcs_phase()
        return self._tf.back_transform_pcs(phase)
