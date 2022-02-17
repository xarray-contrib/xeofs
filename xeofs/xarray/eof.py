from typing import Iterable, Optional, Union

import numpy as np
import xarray as xr

from ..models._eof_base import _EOF_base
from xeofs.xarray._dataarray_transformer import _DataArrayTransformer


class EOF(_EOF_base):
    '''EOF analysis of a single ``xr.DataArray``.

    Parameters
    ----------
    X : xr.DataArray
        Data to be decomposed.
    n_modes : Optional[int]
        Number of modes to compute. Computing less modes can results in
        performance gains. If None, then the maximum number of modes is
        equivalent to ``min(n_samples, n_features)`` (the default is None).
    norm : bool
        Normalize each feature (e.g. grid cell) by its temporal standard
        deviation (the default is False).
    dim : str
        Define the dimension which should considered for maximising variance.
        For most applications in climate science, temporal variance is
        maximised (also known as S-mode EOF analysis) i.e. the time dimension
        should be chosen. If spatial variance should be maximised
        (i.e. T-mode EOF analysis), set e.g. ``dim=['lon', 'lat']``
        (the default is ``time``).

    Examples
    --------

    Import package and create data:

    >>> import xarray as xr
    >>> from xeofs.xarray import EOF
    >>> da = xr.tutorial.load_dataset('rasm')['Tair']
    >>> da = xr.tutorial.load_dataset('air_temperature')['air']
    >>> da = da.isel(lon=slice(0, 3), lat=slice(0, 2))

    Initialize standardized EOF analysis and compute the first 2 modes:

    >>> model = EOF(da, norm=True, n_modes=2)
    >>> model.solve()

    Get explained variance:

    >>> model.explained_variance()
    ... xarray.DataArray'explained_variance'mode: 2
    ...     array([5.8630486 , 0.12335653], dtype=float32)
    ... Coordinates:
    ...     mode (mode) int64 1 2
    ... Attributes:
    ...     (0)

    Get EOFs:

    >>> model.eofs()
    ... xarray.DataArray 'EOFs' lon: 3 lat: 2 mode: 2
    ... array([[[ 0.4083837 , -0.39021498],
    ...         [ 0.40758175,  0.42474997]],
    ...
    ...        [[ 0.40875185, -0.40969774],
    ...         [ 0.40863484,  0.41475943]],
    ...
    ...        [[ 0.40776297, -0.4237887 ],
    ...         [ 0.40837321,  0.38450614]]])
    ... Coordinates:
    ...     lat (lat) float32 75.0 72.5
    ...     lon (lon) float32 200.0 202.5 205.0
    ...     mode (mode) int64 1 2
    ... Attributes:
    ...     (0)

    Get PCs:

    >>> model.pcs()
    ... xarray.DataArray 'PCs' time: 2920 mode: 2
    ... array([[-3.782707  , -0.07754549],
    ...        [-3.7966802 , -0.13775176],
    ...        [-3.7969239 , -0.05770111],
    ...        ...,
    ...        [-3.2584608 ,  0.3592216 ],
    ...        [-3.031799  ,  0.2055658 ],
    ...        [-3.0840495 ,  0.25031802]], dtype=float32)
    ... Coordinates:
    ...     time (time) datetime64[ns] 2013-01-01 ... 2014-12-31T18:00:00
    ...     mode (mode) int64 1 2
    ... Attributes:
    ...     (0)

    '''

    def __init__(
        self,
        X: xr.DataArray,
        n_modes : Optional[int] = None,
        norm : bool = False,
        dim: Union[str, Iterable[str]] = 'time'
    ):

        if(np.logical_not(isinstance(X, xr.DataArray))):
            raise ValueError('This interface is for `xarray.DataArray` only.')

        self._tf = _DataArrayTransformer()
        X = self._tf.fit_transform(X, dim=dim)

        super().__init__(
            X=X,
            n_modes=n_modes,
            norm=norm
        )
        self._mode_idx = xr.IndexVariable('mode', range(1, self.n_modes + 1))
        self._dim = dim

    def singular_values(self) -> xr.DataArray:
        svalues = super().singular_values()
        return xr.DataArray(
            svalues,
            dims=['mode'],
            coords={'mode' : self._mode_idx},
            name='singular_values'
        )

    def explained_variance(self) -> xr.DataArray:
        expvar = super().explained_variance()
        return xr.DataArray(
            expvar,
            dims=['mode'],
            coords={'mode' : self._mode_idx},
            name='explained_variance'
        )

    def explained_variance_ratio(self) -> xr.DataArray:
        expvar = super().explained_variance_ratio()
        return xr.DataArray(
            expvar,
            dims=['mode'],
            coords={'mode' : self._mode_idx},
            name='explained_variance_ratio'
        )

    def eofs(self) -> xr.DataArray:
        eofs = super().eofs()
        eofs = self._tf.back_transform_eofs(eofs)
        eofs.name = 'EOFs'
        return eofs

    def pcs(self) -> xr.DataArray:
        pcs = super().pcs()
        pcs = self._tf.back_transform_pcs(pcs)
        pcs.name = 'PCs'
        return pcs
