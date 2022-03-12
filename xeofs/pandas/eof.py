from typing import Optional, Tuple, Union, List

import pandas as pd

from ..models._base_eof import _BaseEOF
from ..utils.tools import squeeze
from xeofs.pandas._transformer import _MultiDataFrameTransformer

DataFrame = pd.DataFrame
DataFrameList = Union[DataFrame, List[DataFrame]]


class EOF(_BaseEOF):
    '''EOF analysis of a single ``pd.DataFrame``.

    Parameters
    ----------
    X : pd.DataFrame
        Data to be decpomposed.
    axis : int
        Axis along which variance should be maximsed (the default is 0).
    n_modes : Optional[int]
        Number of modes to compute. Computing less modes can results in
        performance gains. If None, then the maximum number of modes is
        equivalent to ``min(n_samples, n_features)`` (the default is None).
    norm : bool
        Normalize each feature (e.g. grid cell) by its temporal standard
        deviation (the default is False).


    Examples
    --------

    Import package and create data:

    >>> import pandas as pd
    >>> from xeofs.pandas import EOF
    >>> rng = np.random.default_rng(7)
    >>> X = rng.standard_normal((4, 3))
    >>> df = pd.DataFrame(X)

    Initialize standardized EOF analysis and compute the first 2 modes:

    >>> model = EOF(df, norm=True, n_modes=2)
    >>> model.solve()

    Get explained variance:

    >>> model.explained_variance()
    ...     	explained_variance
    ...     mode
    ...     1	2.562701
    ...     2	1.167054

    Get EOFs:

    >>> model.eofs()
    ... mode	1	2
    ... 0	0.626041	-0.428431
    ... 1	0.677121	-0.115737
    ... 2	0.386755	0.896132

    Get PCs:

    >>> model.pcs()
    ... mode	1	2
    ... 0	0.495840	-0.221963
    ... 1	-2.254508	-0.470420
    ... 2	1.516900	-0.876695
    ... 3	0.241768	1.569078

    '''

    def __init__(
        self,
        X: DataFrameList,
        axis : int = 0,
        n_modes : Optional[int] = None,
        norm : bool = False,
        weights : Optional[DataFrameList] = None
    ):

        self._tf = _MultiDataFrameTransformer()
        X = self._tf.fit_transform(X)
        weights = self._tf.transform_weights(weights)

        super().__init__(
            X=X,
            n_modes=n_modes,
            norm=norm,
            weights=weights
        )
        self._idx_mode = pd.Index(range(1, self.n_modes + 1), name='mode')

    def singular_values(self) -> DataFrame:
        svalues = super().singular_values()
        svalues = pd.DataFrame(
            svalues,
            columns=['singular_values'],
            index=self._idx_mode
        )
        return svalues

    def explained_variance(self) -> DataFrame:
        expvar = super().explained_variance()
        expvar = pd.DataFrame(
            expvar,
            columns=['explained_variance'],
            index=self._idx_mode
        )
        return expvar

    def explained_variance_ratio(self) -> DataFrame:
        expvar = super().explained_variance_ratio()
        expvar = pd.DataFrame(
            expvar,
            columns=['explained_variance_ratio'],
            index=self._idx_mode
        )
        return expvar

    def eofs(self, scaling : int = 0) -> DataFrameList:
        eofs = super().eofs(scaling=scaling)
        eofs = self._tf.back_transform_eofs(eofs)
        return squeeze(eofs)

    def pcs(self, scaling : int = 0) -> DataFrame:
        pcs = super().pcs(scaling=scaling)
        pcs = self._tf.back_transform_pcs(pcs)
        return pcs

    def eofs_as_correlation(self) -> Tuple[DataFrameList, DataFrameList]:
        corr, pvals = super().eofs_as_correlation()
        corr = self._tf.back_transform_eofs(corr)
        pvals = self._tf.back_transform_eofs(pvals)
        return squeeze(corr), squeeze(pvals)

    def reconstruct_X(
        self,
        mode : Optional[Union[int, List[int], slice]] = None
    ) -> DataFrameList:
        Xrec = super().reconstruct_X(mode)
        Xrec = self._tf.back_transform(Xrec)
        return squeeze(Xrec)

    def project_onto_eofs(
        self,
        X : DataFrameList,
        scaling : int = 0
    ) -> DataFrame:
        '''Project new data onto the EOFs.

        Parameters
        ----------
        X : pd.DataFrame
             New data to project. Data must have same feature shape as original
             data.
        scaling : [0, 1, 2]
            Projections are scaled (i) to be orthonormal (``scaling=0``), (ii) by the
            square root of the eigenvalues (``scaling=1``) or (iii) by the
            singular values (``scaling=2``). In case no weights were applied,
            scaling by the singular values results in the projections having the
            unit of the input data (the default is 0).

        '''
        proj = _MultiDataFrameTransformer()
        X = proj.fit_transform(X, axis=self._tf.axis_samples)
        pcs = super().project_onto_eofs(X=X, scaling=scaling)
        return proj.back_transform_pcs(pcs)
