from typing import Optional
from typing_extensions import Self

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
    ):
        super().__init__()
        self.with_center = with_center
        self.with_std = with_std
        self.with_coslat = with_coslat

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
            self.mean_: DataVar = X.mean(self.sample_dims).compute()

        if params["with_std"]:
            self.std_: DataVar = X.std(self.sample_dims).compute()

        if params["with_coslat"]:
            self.coslat_weights_: DataVar = compute_sqrt_cos_lat_weights(
                data=X, feature_dims=self.feature_dims
            ).compute()

        # Convert None weights to ones
        self.weights_: DataVar = self._process_weights(X, weights).compute()

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


# class DataListScaler(Scaler):
#     """Scale a list of xr.DataArray along sample dimensions.

#     Scaling includes (i) removing the mean and, optionally, (ii) dividing by the standard deviation,
#     (iii) multiplying by the square root of cosine of latitude weights (area weighting; coslat weighting),
#     and (iv) multiplying by additional user-defined weights.

#     Parameters
#     ----------
#     with_std : bool, default=True
#         If True, the data is divided by the standard deviation.
#     with_coslat : bool, default=False
#         If True, the data is multiplied by the square root of cosine of latitude weights.
#     with_weights : bool, default=False
#         If True, the data is multiplied by additional user-defined weights.

#     """

#     def __init__(self, with_std=False, with_coslat=False):
#         super().__init__(with_std=with_std, with_coslat=with_coslat)
#         self.scalers = []

#     def _verify_input(self, data, name: str):
#         """Verify that the input data is a list of DataArrays.

#         Parameters
#         ----------
#         data : list of xarray.DataArray
#             Data to be checked.

#         """
#         if not isinstance(data, list):
#             raise TypeError(f"{name} must be a list of xarray DataArrays or Datasets")
#         if not all(isinstance(da, (xr.DataArray, xr.Dataset)) for da in data):
#             raise TypeError(f"{name} must be a list of xarray DataArrays or Datasets")

#     def fit(
#         self,
#         data: List[Data],
#         sample_dims: Dims,
#         feature_dims_list: DimsList,
#         weights: Optional[List[Data] | Data] = None,
#     ) -> Self:
#         """Fit the scaler to the data.

#         Parameters
#         ----------
#         data : list of xarray.DataArray
#             Data to be scaled.
#         sample_dims : hashable or sequence of hashable
#             Dimensions along which the data is considered to be a sample.
#         feature_dims_list : list of hashable or list of sequence of hashable
#             List of dimensions along which the data is considered to be a feature.
#         weights : list of xarray.DataArray, optional
#             List of weights to be applied to the data. Must have the same dimensions as the data.

#         """
#         self._verify_input(data, "data")

#         # Check input
#         if not isinstance(feature_dims_list, list):
#             err_message = "feature dims must be a list of the feature dimensions of each DataArray, "
#             err_message += 'e.g. [("lon", "lat"), ("lon")]'
#             raise TypeError(err_message)

#         # Sample dimensions are the same for all data arrays
#         # Feature dimensions may be different for each data array
#         self.dims = {"sample": sample_dims, "feature": feature_dims_list}

#         # However, for each DataArray a list of feature dimensions must be provided
#         _check_parameter_number("feature_dims", feature_dims_list, len(data))

#         # If no weights are provided, create a list of None
#         self.weights = process_parameter("weights", weights, None, len(data))

#         params = self.get_params()

#         for da, wghts, fdims in zip(data, self.weights, feature_dims_list):
#             # Create Scaler object for each data array
#             scaler = Scaler(**params)
#             scaler.fit(da, sample_dims=sample_dims, feature_dims=fdims, weights=wghts)
#             self.scalers.append(scaler)

#         return self

#     def transform(self, da_list: List[Data]) -> List[Data]:
#         """Scale the data.

#         Parameters
#         ----------
#         da_list : list of xarray.DataArray
#             Data to be scaled.

#         Returns
#         -------
#         list of xarray.DataArray
#             Scaled data.

#         """
#         self._verify_input(da_list, "da_list")

#         da_list_transformed = []
#         for scaler, da in zip(self.scalers, da_list):
#             da_list_transformed.append(scaler.transform(da))
#         return da_list_transformed

#     def fit_transform(
#         self,
#         data: List[Data],
#         sample_dims: Dims,
#         feature_dims_list: DimsList,
#         weights: Optional[List[Data] | Data] = None,
#     ) -> List[Data]:
#         """Fit the scaler to the data and scale it.

#         Parameters
#         ----------
#         data : list of xr.DataArray
#             Data to be scaled.
#         sample_dims : hashable or sequence of hashable
#             Dimensions along which the data is considered to be a sample.
#         feature_dims_list : list of hashable or list of sequence of hashable
#             List of dimensions along which the data is considered to be a feature.
#         weights : list of xr.DataArray, optional
#             List of weights to be applied to the data. Must have the same dimensions as the data.

#         Returns
#         -------
#         list of xarray.DataArray
#             Scaled data.

#         """
#         self.fit(data, sample_dims, feature_dims_list, weights)
#         return self.transform(data)

#     def inverse_transform_data(self, da_list: List[Data]) -> List[Data]:
#         """Unscale the data.

#         Parameters
#         ----------
#         da_list : list of xarray.DataArray
#             Data to be scaled.

#         Returns
#         -------
#         list of xarray.DataArray
#             Scaled data.

#         """
#         self._verify_input(da_list, "da_list")

#         da_list_transformed = []
#         for scaler, da in zip(self.scalers, da_list):
#             da_list_transformed.append(scaler.inverse_transform_data(da))
#         return da_list_transformed

#     def inverse_transform_components(self, da_list: List[Data]) -> List[Data]:
#         return da_list
