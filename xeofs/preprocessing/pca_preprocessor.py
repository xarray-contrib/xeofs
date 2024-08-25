from typing import List, Optional, Tuple

import numpy as np
from typing_extensions import Self

from ..utils.data_types import (
    Data,
    Dims,
    DimsList,
)
from .concatenator import Concatenator
from .dimension_renamer import DimensionRenamer
from .multi_index_converter import MultiIndexConverter
from .preprocessor import Preprocessor
from .sanitizer import Sanitizer
from .scaler import Scaler
from .stacker import Stacker
from .whitener import Whitener


def extract_new_dim_names(X: List[DimensionRenamer]) -> Tuple[Dims, DimsList]:
    """Extract the new dimension names from a list of DimensionRenamer objects.

    Parameters
    ----------
    X : list of DimensionRenamer
        List of DimensionRenamer objects.

    Returns
    -------
    Dims
        Sample dimensions
    DimsList
        Feature dimenions

    """
    new_sample_dims = []
    new_feature_dims: DimsList = []
    for x in X:
        new_sample_dims.append(x.sample_dims_after)
        new_feature_dims.append(x.feature_dims_after)
    new_sample_dims: Dims = tuple(np.unique(np.asarray(new_sample_dims)).tolist())
    return new_sample_dims, new_feature_dims


class PCAPreprocessor(Preprocessor):
    """Preprocess xarray objects and transform into (whitened) PC space.

    PCA-Preprocesser includes steps from Preprocessor class:
        (i) Feature-wise scaling (e.g. removing mean, dividing by standard deviation, applying (latitude) weights
        (ii) Renaming dimensions (to avoid conflicts with sample and feature dimensions)
        (iii) Converting MultiIndexes to regular Indexes (MultiIndexes cannot be stacked)
        (iv) Stacking the data into 2D DataArray
        (v) Converting MultiIndexes introduced by stacking into regular Indexes
        (vi) Removing NaNs
        (vii) Concatenating the 2D DataArrays into one 2D DataArray
        (viii) Transform into whitened PC space


    Parameters
    ----------
    n_modes : int | float
        If int, specifies the number of modes to retain. If float, specifies the fraction of variance in the whitened data that should be explained by the retained modes.
    alpha : float, default=0.0
        Degree of whitening. If 0, the data is completely whitened. If 1, the data is not whitened.
    init_rank_reduction : float, default=0.3
        Fraction of the initial rank to reduce the data to before applying PCA.
    sample_name : str, default="sample"
        Name of the sample dimension.
    feature_name : str, default="feature"
        Name of the feature dimension.
    with_center : bool, default=True
        If True, the data is centered by subtracting the mean.
    with_std : bool, default=True
        If True, the data is divided by the standard deviation.
    with_coslat : bool, default=False
        If True, the data is multiplied by the square root of cosine of latitude weights.
    with_weights : bool, default=False
        If True, the data is multiplied by additional user-defined weights.
    return_list : bool, default=True
        If True, inverse_transform methods returns always a list of DataArray(s).
        If False, the output is returned as a single DataArray if possible.
    check_nans : bool, default=True
        If True, remove full-dimensional NaN features from the data, check to ensure
        that NaN features match the original fit data during transform, and check
        for isolated NaNs. Note: this forces eager computation of dask arrays.
        If False, skip all NaN checks. In this case, NaNs should be explicitly removed
        or filled prior to fitting, or SVD will fail.

    """

    def __init__(
        self,
        n_modes: int | float,
        alpha: float = 1.0,
        init_rank_reduction: float = 0.3,
        sample_name: str = "sample",
        feature_name: str = "feature",
        with_center: bool = True,
        with_std: bool = False,
        with_coslat: bool = False,
        return_list: bool = True,
        check_nans: bool = True,
        compute: bool = True,
    ):
        super().__init__(
            sample_name=sample_name,
            feature_name=feature_name,
            with_center=with_center,
            with_std=with_std,
            with_coslat=with_coslat,
            return_list=return_list,
            check_nans=check_nans,
            compute=compute,
        )

        # Set parameters
        self.n_modes = n_modes
        self.alpha = alpha
        self.init_rank_reduction = init_rank_reduction

        # 8 | PCA-Whitener
        self.whitener = Whitener(
            n_modes=self.n_modes,
            init_rank_reduction=self.init_rank_reduction,
            alpha=self.alpha,
            sample_name=self.sample_name,
            feature_name=self.feature_name,
        )

    def transformer_types(self):
        """Ordered list of transformer operations."""
        return dict(
            scaler=Scaler,
            renamer=DimensionRenamer,
            preconverter=MultiIndexConverter,
            stacker=Stacker,
            postconverter=MultiIndexConverter,
            sanitizer=Sanitizer,
            concatenator=Concatenator,
            whitener=Whitener,
        )

    def _fit_algorithm(
        self,
        X: List[Data] | Data,
        sample_dims: Dims,
        weights: Optional[List[Data] | Data] = None,
    ) -> Tuple[Self, Data]:
        _, X = super()._fit_algorithm(X, sample_dims, weights)

        # 8 | PCA-Whitening
        X = self.whitener.fit_transform(X)  # type: ignore

        return self, X
