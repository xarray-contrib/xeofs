from typing import Iterable, Optional, Union, Tuple, List

import numpy as np
import xarray as xr

from ..models._base_eof import _BaseEOF
from ._dataarray_transformer import _DataArrayTransformer


class EOF(_BaseEOF):
    '''EOF analysis of a single ``xr.DataArray``.

    Parameters
    ----------
    X : xr.DataArray
        Data to be decomposed.
    dim : str
        Define the dimension which should considered for maximising variance.
        For most applications in climate science, temporal variance is
        maximised (also known as S-mode EOF analysis) i.e. the time dimension
        should be chosen. If spatial variance should be maximised
        (i.e. T-mode EOF analysis), set e.g. ``dim=['lon', 'lat']``
        (the default is ``time``).
    n_modes : Union[int | None]
        Number of modes to compute. Computing less modes can results in
        performance gains. If None, then the maximum number of modes is
        equivalent to ``min(n_samples, n_features)`` (the default is None).
    norm : bool
        Normalize each feature (e.g. grid cell) by its temporal standard
        deviation (the default is False).
    weights : Union[xr.DatArray | str | None]
        Weights to be applied to data (features).


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
        dim: Union[str, Iterable[str]] = 'time',
        n_modes : Optional[int] = None,
        norm : bool = False,
        weights : Optional[Union[xr.DataArray, str]] = None
    ):

        self._tf = _DataArrayTransformer()
        self._tf.fit(X, dim=dim)
        if weights == 'coslat':
            weights = self._get_coslat_weights(X)
        X = self._tf.transform(X)
        weights = self._tf.transform_weights(weights)

        super().__init__(
            X=X,
            n_modes=n_modes,
            norm=norm,
            weights=weights
        )
        self._idx_mode = xr.IndexVariable('mode', range(1, self.n_modes + 1))
        self._dim = dim

    def _get_coslat_weights(self, X : xr.DataArray) -> xr.DataArray:
        # Find dimension name of latitude
        possible_lat_names = [
            'latitude', 'Latitude', 'lat', 'Lat', 'LATITUDE', 'LAT'
        ]
        idx_lat_dim = np.isin(X.dims, possible_lat_names)
        try:
            lat_dim = np.array(X.dims)[idx_lat_dim][0]
        except IndexError:
            err_msg = (
                'Latitude dimension cannot be found. Please make sure '
                'latitude dimensions is called like one of {:}'
            )
            err_msg = err_msg.format(possible_lat_names)
            raise ValueError(err_msg)
        # Check if latitude is a MultiIndex => not allowed
        if X.coords[lat_dim].dtype not in [np.float_, np.float64, np.float32, np.int_]:
            err_msg = 'MultiIndex as latitude dimensions is not allowed.'
            raise ValueError(err_msg)
        # Compute coslat weights
        weights = np.cos(np.deg2rad(X.coords[lat_dim]))
        weights = np.sqrt(weights.where(weights > 0, 0))
        # Broadcast latitude weights on other feature dimensions
        sample_dims = self._tf.dims_samples
        feature_grid = X.isel({k: 0 for k in sample_dims})
        feature_grid = feature_grid.drop_vars(sample_dims)
        return weights.broadcast_like(feature_grid)

    def singular_values(self) -> xr.DataArray:
        svalues = super().singular_values()
        return xr.DataArray(
            svalues,
            dims=['mode'],
            coords={'mode' : self._idx_mode},
            name='singular_values'
        )

    def explained_variance(self) -> xr.DataArray:
        expvar = super().explained_variance()
        return xr.DataArray(
            expvar,
            dims=['mode'],
            coords={'mode' : self._idx_mode},
            name='explained_variance'
        )

    def explained_variance_ratio(self) -> xr.DataArray:
        expvar = super().explained_variance_ratio()
        return xr.DataArray(
            expvar,
            dims=['mode'],
            coords={'mode' : self._idx_mode},
            name='explained_variance_ratio'
        )

    def eofs(self, scaling : int = 0) -> xr.DataArray:
        eofs = super().eofs(scaling=scaling)
        eofs = self._tf.back_transform_eofs(eofs)
        eofs.name = 'EOFs'
        return eofs

    def pcs(self, scaling : int = 0) -> xr.DataArray:
        pcs = super().pcs(scaling=scaling)
        pcs = self._tf.back_transform_pcs(pcs)
        pcs.name = 'PCs'
        return pcs

    def eofs_as_correlation(self) -> Tuple[xr.DataArray, xr.DataArray]:
        corr, pvals = super().eofs_as_correlation()
        corr = self._tf.back_transform_eofs(corr)
        pvals = self._tf.back_transform_eofs(pvals)
        corr.name = 'correlation_coeffient'
        pvals.name = 'p_value'
        return corr, pvals

    def reconstruct_X(
        self,
        mode : Optional[Union[int, List[int], slice]] = None
    ) -> xr.DataArray:
        Xrec = super().reconstruct_X(mode=mode)
        Xrec = self._tf.back_transform(Xrec)
        coords = {dim: self._tf.coords[dim] for dim in self._tf.dims_samples}
        Xrec = Xrec.assign_coords(coords)
        Xrec.name = 'X_reconstructed'
        return Xrec

    def project_onto_eofs(
        self,
        X : xr.DataArray,
        scaling : int = 0
    ) -> xr.DataArray:
        '''Project new data onto the EOFs.

        Parameters
        ----------
        X : xr.DataArray
             New data to project. Data must have same feature shape as original
             data.
        scaling : [0, 1, 2]
            Projections are scaled (i) to be orthonormal (``scaling=0``), (ii) by the
            square root of the eigenvalues (``scaling=1``) or (iii) by the
            singular values (``scaling=2``). In case no weights were applied,
            scaling by the singular values results in the projections having the
            unit of the input data (the default is 0).

        '''
        proj = _DataArrayTransformer()
        X = proj.fit_transform(X, dim=self._tf.dims_samples)
        pcs = super().project_onto_eofs(X=X, scaling=scaling)
        return proj.back_transform_pcs(pcs)
