from typing import Optional, Tuple, List, Union

import pandas as pd

from ..models._base_rock_pca import _BaseROCK_PCA
from ..utils.tools import squeeze
from xeofs.pandas._transformer import _MultiDataFrameTransformer

DataFrame = pd.DataFrame
DataFrameList = Union[DataFrame, List[DataFrame]]


class ROCK_PCA(_BaseROCK_PCA):
    '''ROCK-PCA of a single or multiple ``pd.DataFrame``.'''

    def __init__(
        self,
        X: DataFrame,
        n_rot : int,
        power : int,
        sigma : float,
        axis : int = 0,
        n_modes: Optional[int] = None,
        norm: bool = False,
        weights: Optional[DataFrame] = None,
    ):
        self._tf = _MultiDataFrameTransformer()
        X = self._tf.fit_transform(X)
        weights = self._tf.transform_weights(weights)

        super().__init__(
            X=X, n_rot=n_rot, power=power, sigma=sigma, n_modes=n_modes,
            norm=norm, weights=weights
        )
        self._idx_mode = pd.Index(range(1, self._params['n_modes'] + 1), name='mode')

    def explained_variance(self) -> DataFrame:
        expvar = super().explained_variance()
        return pd.DataFrame(
            expvar,
            columns=['explained_variance'],
            index=self._idx_mode
        )

    def explained_variance_ratio(self) -> DataFrame:
        expvar_ratio = super().explained_variance_ratio()
        return pd.DataFrame(
            expvar_ratio,
            columns=['explained_variance_ratio'],
            index=self._idx_mode
        )

    def eofs(self) -> DataFrameList:
        eofs = super().eofs()
        eofs = self._tf.back_transform_eofs(eofs)
        return squeeze(eofs)

    def pcs(self) -> DataFrame:
        pcs = super().pcs()
        return self._tf.back_transform_pcs(pcs)

    def eofs_amplitude(self) -> DataFrameList:
        amp = super().eofs_amplitude()
        amp = self._tf.back_transform_eofs(amp)
        return squeeze(amp)

    def pcs_amplitude(self) -> DataFrame:
        amp = super().pcs_amplitude()
        return self._tf.back_transform_pcs(amp)

    def eofs_phase(self) -> DataFrameList:
        phase = super().eofs_phase()
        phase = self._tf.back_transform_eofs(phase)
        return squeeze(phase)

    def pcs_phase(self) -> DataFrame:
        phase = super().pcs_phase()
        return self._tf.back_transform_pcs(phase)
