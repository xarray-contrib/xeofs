from typing import Optional, Dict
from typing_extensions import Self

import dask
import numpy as np
import xarray as xr

from .transformer import Transformer
from ..utils.data_types import Dims, DataArray, DataSet, Data, DataVar, DataVarBound
from ..utils.xarray_utils import compute_sqrt_cos_lat_weights, feature_ones_like


class Scaler(Transformer):
    """Scale the data along sample dimensions.

    Scaling includes (i) removing the mean and, optionally, (ii) dividing by the standard deviation,
    (iii) multiplying by the square root of cosine of latitude weights (area weighting; coslat weighting),
    and (iv) multiplying by additional user-defined weights.

    Parameters
    ----------
    with_center : bool, default=True
        If True, the data is centered by subtracting the mean.
    with_std : bool, default=True
        If True, the data is divided by the standard deviation.
    with_coslat : bool, default=False
        If True, the data is multiplied by the square root of cosine of latitude weights.
    weights : DataArray | Dataset, optional
        Weights to be applied to the data. Must have the same dimensions as the data.
        If None, no weights are applied.
    """

    def __init__(
        self,
        with_center: bool = True,
        with_std: bool = False,
        with_coslat: bool = False,
        compute: bool = True,
    ):
        super().__init__()
        self.with_center = with_center
        self.with_std = with_std
        self.with_coslat = with_coslat
        self.compute = compute

        self.mean_ = xr.DataArray(name="mean_")
        self.std_ = xr.DataArray(name="std_")
        self.coslat_weights_ = xr.DataArray(name="coslat_weights_")
        self.weights_ = xr.DataArray(name="weights_")

    def get_serialization_attrs(self) -> Dict:
        return dict(
            mean_=self.mean_,
            std_=self.std_,
            coslat_weights_=self.coslat_weights_,
            weights_=self.weights_,
        )

    def _verify_input(self, X, name: str):
        if not isinstance(X, (xr.DataArray, xr.Dataset)):
            raise TypeError(f"{name} must be an xarray DataArray or Dataset")

    def _process_weights(self, X: DataVarBound, weights) -> DataVarBound:
        if weights is None:
            wghts: DataVarBound = feature_ones_like(X, self.feature_dims)
        else:
            wghts: DataVarBound = weights

        return wghts

    def fit(
        self,
        X: DataVar,
        sample_dims: Dims,
        feature_dims: Dims,
        weights: Optional[DataVar] = None,
    ) -> Self:
        """Fit the scaler to the data.

        Parameters
        ----------
        X : DataArray | Dataset
            Data to be scaled.
        sample_dims : sequence of hashable
            Dimensions along which the data is considered to be a sample.
        feature_dims : sequence of hashable
            Dimensions along which the data is considered to be a feature.
        weights : DataArray | Dataset, optional
            Weights to be applied to the data. Must have the same dimensions as the data.
            If None, no weights are applied.

        """
        # Check input types
        self._verify_input(X, "data")

        self.sample_dims = sample_dims
        self.feature_dims = feature_dims
        # Store sample and feature dimensions for later use
        self.dims = {"sample": sample_dims, "feature": feature_dims}

        params = self.get_params()

        # Scaling parameters are computed along sample dimensions
        if params["with_center"]:
            self.mean_: DataVar = X.mean(self.sample_dims)

        if params["with_std"]:
            self.std_: DataVar = X.std(self.sample_dims).clip(
                min=np.finfo(np.float32).eps
            )

        if params["with_coslat"]:
            self.coslat_weights_: DataVar = compute_sqrt_cos_lat_weights(
                data=X, feature_dims=self.feature_dims
            )

        # Convert None weights to ones
        self.weights_: DataVar = self._process_weights(X, weights)

        if self.get_params()["compute"]:
            (self.mean_, self.std_, self.coslat_weights_, self.weights_) = dask.compute(
                self.mean_,
                self.std_,
                self.coslat_weights_,
                self.weights_,
            )

        return self

    def transform(self, X: DataVarBound) -> DataVarBound:
        """Scale the data.

        Parameters
        ----------
        data : DataArray | Dataset
            Data to be scaled.

        Returns
        -------
        DataArray | Dataset
            Scaled data.

        """
        self._verify_input(X, "X")

        params = self.get_params()

        if params["with_center"]:
            X = X - self.mean_
        if params["with_std"]:
            X = X / self.std_
        if params["with_coslat"]:
            X = X * self.coslat_weights_

        X = X * self.weights_
        return X

    def fit_transform(
        self,
        X: DataVarBound,
        sample_dims: Dims,
        feature_dims: Dims,
        weights: Optional[DataVarBound] = None,
    ) -> DataVarBound:
        return self.fit(X, sample_dims, feature_dims, weights).transform(X)

    def inverse_transform_data(self, X: DataVarBound) -> DataVarBound:
        """Unscale the data.

        Parameters
        ----------
        X : DataArray | DataSet
            Data to be unscaled.

        Returns
        -------
        DataArray | DataSet
            Unscaled data.

        """
        self._verify_input(X, "X")

        params = self.get_params()
        X = X / self.weights_
        if params["with_coslat"]:
            X = X / self.coslat_weights_
        if params["with_std"]:
            X = X * self.std_
        if params["with_center"]:
            X = X + self.mean_

        return X

    def inverse_transform_components(self, X: DataVarBound) -> DataVarBound:
        return X

    def inverse_transform_scores(self, X: DataArray) -> DataArray:
        return X

    def inverse_transform_scores_unseen(self, X: DataArray) -> DataArray:
        return X
