from typing import List, Optional, Sequence, Hashable, Self

import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils.sanity_checks import (
    assert_single_dataset,
    assert_list_dataarrays,
    convert_to_dim_type,
)
from ..utils.data_types import (
    Dims,
    DimsList,
    DataArray,
    DataSet,
    DataList,
)
from ..utils.xarray_utils import compute_sqrt_cos_lat_weights


class DataArrayScaler(BaseEstimator, TransformerMixin):
    """Scale the data along sample dimensions.

    Scaling includes (i) removing the mean and, optionally, (ii) dividing by the standard deviation,
    (iii) multiplying by the square root of cosine of latitude weights (area weighting; coslat weighting),
    and (iv) multiplying by additional user-defined weights.

    Parameters
    ----------
    with_std : bool, default=True
        If True, the data is divided by the standard deviation.
    with_coslat : bool, default=False
        If True, the data is multiplied by the square root of cosine of latitude weights.
    with_weights : bool, default=False
        If True, the data is multiplied by additional user-defined weights.

    """

    def __init__(self, with_std=False, with_coslat=False, with_weights=False):
        self.with_std = with_std
        self.with_coslat = with_coslat
        self.with_weights = with_weights

    def _verify_input(self, data: DataArray, name: str):
        if not isinstance(data, xr.DataArray):
            raise ValueError(f"{name} must be an xarray DataArray")

    def _compute_sqrt_cos_lat_weights(self, data, dim):
        """Compute the square root of cosine of latitude weights.

        Parameters
        ----------
        data : DataArray | DataSet
            Data to be scaled.
        dim : sequence of hashable
            Dimensions along which the data is considered to be a feature.

        Returns
        -------
        DataArray | DataSet
            Square root of cosine of latitude weights.

        """
        self._verify_input(data, "data")

        weights = compute_sqrt_cos_lat_weights(data, dim)
        weights.name = "coslat_weights"

        return weights

    def fit(
        self,
        data: DataArray,
        sample_dims: Dims,
        feature_dims: Dims,
        weights: Optional[DataArray] = None,
    ) -> Self:
        """Fit the scaler to the data.

        Parameters
        ----------
        data : DataArray
            Data to be scaled.
        sample_dims : sequence of hashable
            Dimensions along which the data is considered to be a sample.
        feature_dims : sequence of hashable
            Dimensions along which the data is considered to be a feature.
        weights : DataArray, optional
            Weights to be applied to the data. Must have the same dimensions as the data.
            If None, no weights are applied.

        """
        # Check input types
        self._verify_input(data, "data")
        if weights is not None:
            self._verify_input(weights, "weights")

        sample_dims = convert_to_dim_type(sample_dims)
        feature_dims = convert_to_dim_type(feature_dims)

        # Store sample and feature dimensions for later use
        self.dims_ = {"sample": sample_dims, "feature": feature_dims}

        # Scaling parameters are computed along sample dimensions
        self.mean_: DataArray = data.mean(sample_dims).compute()

        params = self.get_params()
        if params["with_std"]:
            self.std_: DataArray = data.std(sample_dims).compute()

        if params["with_coslat"]:
            self.coslat_weights_: DataArray = self._compute_sqrt_cos_lat_weights(
                data, feature_dims
            ).compute()

        if params["with_weights"]:
            if weights is None:
                raise ValueError("Weights must be provided when with_weights is True")
            self.weights_: DataArray = weights.compute()

        return self

    def transform(self, data: DataArray) -> DataArray:
        """Scale the data.

        Parameters
        ----------
        data : DataArray
            Data to be scaled.

        Returns
        -------
        DataArray
            Scaled data.

        """
        self._verify_input(data, "data")

        data = data - self.mean_

        params = self.get_params()
        if params["with_std"]:
            data = data / self.std_
        if params["with_coslat"]:
            data = data * self.coslat_weights_
        if params["with_weights"]:
            data = data * self.weights_
        return data

    def fit_transform(
        self,
        data: DataArray,
        sample_dims: Dims,
        feature_dims: Dims,
        weights: Optional[DataArray] = None,
    ) -> DataArray:
        """Fit the scaler to the data and scale it.

        Parameters
        ----------
        data : DataArray
            Data to be scaled.
        sample_dims : sequence of hashable
            Dimensions along which the data is considered to be a sample.
        feature_dims : sequence of hashable
            Dimensions along which the data is considered to be a feature.
        weights : DataArray, optional
            Weights to be applied to the data. Must have the same dimensions as the data.
            If None, no weights are applied.

        Returns
        -------
        DataArray
            Scaled data.

        """

        return self.fit(data, sample_dims, feature_dims, weights).transform(data)

    def inverse_transform_data(self, data: DataArray) -> DataArray:
        """Unscale the data.

        Parameters
        ----------
        data : DataArray | DataSet
            Data to be unscaled.

        Returns
        -------
        DataArray | DataSet
            Unscaled data.

        """
        self._verify_input(data, "data")

        params = self.get_params()
        if params["with_weights"]:
            data = data / self.weights_
        if params["with_coslat"]:
            data = data / self.coslat_weights_
        if params["with_std"]:
            data = data * self.std_

        data = data + self.mean_

        return data

    def inverse_transform_components(self, data: DataArray) -> DataArray:
        return data

    def inverse_transform_scores(self, data: DataArray) -> DataArray:
        return data


class DataSetScaler(DataArrayScaler):
    def _verify_input(self, data: DataSet, name: str):
        """Verify that the input data is a Dataset.

        Parameters
        ----------
        data : xarray.Dataset
            Data to be checked.

        """
        assert_single_dataset(data, name)

    def _compute_sqrt_cos_lat_weights(self, data: DataSet, dim) -> DataArray:
        return super()._compute_sqrt_cos_lat_weights(data, dim)

    def fit(
        self,
        data: DataSet,
        sample_dims: Hashable | Sequence[Hashable],
        feature_dims: Hashable | Sequence[Hashable],
        weights: Optional[DataSet] = None,
    ) -> Self:
        return super().fit(data, sample_dims, feature_dims, weights)  # type: ignore

    def transform(self, data: DataSet) -> DataSet:
        return super().transform(data)  # type: ignore

    def fit_transform(
        self,
        data: DataSet,
        sample_dims: Hashable | Sequence[Hashable],
        feature_dims: Hashable | Sequence[Hashable],
        weights: Optional[DataSet] = None,
    ) -> DataSet:
        return super().fit_transform(data, sample_dims, feature_dims, weights)  # type: ignore

    def inverse_transform_data(self, data: DataSet) -> DataSet:
        return super().inverse_transform_data(data)  # type: ignore

    def inverse_transform_components(self, data: DataSet) -> DataSet:
        return super().inverse_transform_components(data)  # type: ignore


class DataListScaler(DataArrayScaler):
    """Scale a list of xr.DataArray along sample dimensions.

    Scaling includes (i) removing the mean and, optionally, (ii) dividing by the standard deviation,
    (iii) multiplying by the square root of cosine of latitude weights (area weighting; coslat weighting),
    and (iv) multiplying by additional user-defined weights.

    Parameters
    ----------
    with_std : bool, default=True
        If True, the data is divided by the standard deviation.
    with_coslat : bool, default=False
        If True, the data is multiplied by the square root of cosine of latitude weights.
    with_weights : bool, default=False
        If True, the data is multiplied by additional user-defined weights.

    """

    def __init__(self, with_std=False, with_coslat=False, with_weights=False):
        super().__init__(
            with_std=with_std, with_coslat=with_coslat, with_weights=with_weights
        )
        self.scalers = []

    def _verify_input(self, data: DataList, name: str):
        """Verify that the input data is a list of DataArrays.

        Parameters
        ----------
        data : list of xarray.DataArray
            Data to be checked.

        """
        assert_list_dataarrays(data, name)

    def fit(
        self,
        data: DataList,
        sample_dims: Dims,
        feature_dims_list: DimsList,
        weights: Optional[DataList] = None,
    ) -> Self:
        """Fit the scaler to the data.

        Parameters
        ----------
        data : list of xarray.DataArray
            Data to be scaled.
        sample_dims : hashable or sequence of hashable
            Dimensions along which the data is considered to be a sample.
        feature_dims_list : list of hashable or list of sequence of hashable
            List of dimensions along which the data is considered to be a feature.
        weights : list of xarray.DataArray, optional
            List of weights to be applied to the data. Must have the same dimensions as the data.

        """
        self._verify_input(data, "data")

        # Check input
        if not isinstance(feature_dims_list, list):
            err_message = "feature dims must be a list of the feature dimensions of each DataArray, "
            err_message += 'e.g. [("lon", "lat"), ("lon")]'
            raise TypeError(err_message)

        sample_dims = convert_to_dim_type(sample_dims)
        feature_dims = [convert_to_dim_type(fdims) for fdims in feature_dims_list]

        # Sample dimensions are the same for all data arrays
        # Feature dimensions may be different for each data array
        self.dims = {"sample": sample_dims, "feature": feature_dims}

        # However, for each DataArray a list of feature dimensions must be provided
        if len(data) != len(feature_dims):
            err_message = (
                "Number of data arrays and feature dimensions must be the same. "
            )
            err_message += f"Got {len(data)} data arrays and {len(feature_dims)} feature dimensions"
            raise ValueError(err_message)

        # If no weights are provided, create a list of None
        if weights is None:
            self.weights = [None] * len(data)
        else:
            self.weights = weights

        # Check that number of weights is the same as number of data arrays
        params = self.get_params()
        if params["with_weights"]:
            if len(data) != len(self.weights):
                err_message = "Number of data arrays and weights must be the same. "
                err_message += (
                    f"Got {len(data)} data arrays and {len(self.weights)} weights"
                )
                raise ValueError(err_message)

        for da, wghts, fdims in zip(data, self.weights, feature_dims):
            # Create DataArrayScaler object for each data array
            scaler = DataArrayScaler(**params)
            scaler.fit(da, sample_dims=sample_dims, feature_dims=fdims, weights=wghts)
            self.scalers.append(scaler)

        return self

    def transform(self, da_list: DataList) -> DataList:
        """Scale the data.

        Parameters
        ----------
        da_list : list of xarray.DataArray
            Data to be scaled.

        Returns
        -------
        list of xarray.DataArray
            Scaled data.

        """
        self._verify_input(da_list, "da_list")

        da_list_transformed = []
        for scaler, da in zip(self.scalers, da_list):
            da_list_transformed.append(scaler.transform(da))
        return da_list_transformed

    def fit_transform(
        self,
        data: DataList,
        sample_dims: Dims,
        feature_dims_list: DimsList,
        weights: Optional[DataList] = None,
    ) -> DataList:
        """Fit the scaler to the data and scale it.

        Parameters
        ----------
        data : list of xr.DataArray
            Data to be scaled.
        sample_dims : hashable or sequence of hashable
            Dimensions along which the data is considered to be a sample.
        feature_dims_list : list of hashable or list of sequence of hashable
            List of dimensions along which the data is considered to be a feature.
        weights : list of xr.DataArray, optional
            List of weights to be applied to the data. Must have the same dimensions as the data.

        Returns
        -------
        list of xarray.DataArray
            Scaled data.

        """
        self.fit(data, sample_dims, feature_dims_list, weights)
        return self.transform(data)

    def inverse_transform_data(self, da_list: DataList) -> DataList:
        """Unscale the data.

        Parameters
        ----------
        da_list : list of xarray.DataArray
            Data to be scaled.

        Returns
        -------
        list of xarray.DataArray
            Scaled data.

        """
        self._verify_input(da_list, "da_list")

        da_list_transformed = []
        for scaler, da in zip(self.scalers, da_list):
            da_list_transformed.append(scaler.inverse_transform_data(da))
        return da_list_transformed

    def inverse_transform_components(self, da_list: DataList) -> DataList:
        return da_list
