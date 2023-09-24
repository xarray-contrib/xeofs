from typing import Optional, Sequence, Hashable, List

from .factory import StackerFactory, ScalerFactory, MultiIndexConverterFactory
from .sanitizer import DataArraySanitizer
from ..utils.xarray_utils import get_dims
from ..utils.data_types import DataObject, DataArray


class Preprocessor:
    """Scale and stack the data along sample dimensions.

    Scaling includes (i) removing the mean and, optionally, (ii) dividing by the standard deviation,
    (iii) multiplying by the square root of cosine of latitude weights (area weighting; coslat weighting),
    and (iv) multiplying by additional user-defined weights.

    Stacking includes (i) stacking the data along the sample dimensions and (ii) stacking the data along the feature dimensions.

    Parameters
    ----------
    with_std : bool, default=True
        If True, the data is divided by the standard deviation.
    with_coslat : bool, default=False
        If True, the data is multiplied by the square root of cosine of latitude weights.
    with_weights : bool, default=False
        If True, the data is multiplied by additional user-defined weights.

    """

    def __init__(
        self,
        sample_name="sample",
        feature_name="feature",
        with_std=True,
        with_coslat=False,
        with_weights=False,
    ):
        self.sample_name = sample_name
        self.feature_name = feature_name
        self.with_std = with_std
        self.with_coslat = with_coslat
        self.with_weights = with_weights

    def fit(
        self,
        data: DataObject,
        dim: Hashable | Sequence[Hashable] | List[Sequence[Hashable]],
        weights: Optional[DataObject] = None,
    ):
        # Set sample and feature dimensions
        sample_dims, feature_dims = get_dims(data, sample_dims=dim)
        self.dims = {self.sample_name: sample_dims, self.feature_name: feature_dims}

        # Create Scaler
        scaler_params = {
            "with_std": self.with_std,
            "with_coslat": self.with_coslat,
            "with_weights": self.with_weights,
        }
        self.scaler = ScalerFactory.create_scaler(data, **scaler_params)
        data = self.scaler.fit_transform(data, sample_dims, feature_dims, weights)

        # Create MultiIndexConverter (Pre)
        self.preconverter = MultiIndexConverterFactory.create_converter(data)
        data = self.preconverter.fit_transform(data)

        # Create Stacker
        stacker_kwargs = {
            "sample_name": self.sample_name,
            "feature_name": self.feature_name,
        }
        self.stacker = StackerFactory.create_stacker(data, **stacker_kwargs)
        data: DataArray = self.stacker.fit_transform(data, sample_dims, feature_dims)

        # Create MultiIndexConverter (Post)
        self.postconverter = MultiIndexConverterFactory.create_converter(data)
        data = self.postconverter.fit_transform(data)

        # Create Sanitizer
        self.sanitizer = DataArraySanitizer(
            sample_name=self.sample_name, feature_name=self.feature_name
        )
        self.sanitizer.fit(data)
        return self

    def transform(self, data: DataObject) -> DataArray:
        data = self.scaler.transform(data)
        data = self.preconverter.transform(data)
        data = self.stacker.transform(data)
        data = self.postconverter.transform(data)
        return self.sanitizer.transform(data)

    def fit_transform(
        self,
        data: DataObject,
        dim: Hashable | Sequence[Hashable] | List[Sequence[Hashable]],
        weights: Optional[DataObject] = None,
    ) -> DataArray:
        return self.fit(data, dim, weights).transform(data)

    def inverse_transform_data(self, data: DataArray) -> DataObject:
        """Inverse transform the data.

        Parameters:
        -------------
        data: xr.DataArray
            Input data.

        Returns
        -------
        xr.DataArray or xr.Dataset or list of xr.DataArray
            The inverse transformed data.

        """
        data = self.sanitizer.inverse_transform_data(data)
        data = self.postconverter.inverse_transform_data(data)
        data = self.stacker.inverse_transform_data(data)
        data = self.preconverter.inverse_transform_data(data)
        return self.scaler.inverse_transform_data(data)

    def inverse_transform_components(self, data: DataArray) -> DataObject:
        """Inverse transform the components.

        Parameters:
        -------------
        data: xr.DataArray or list of xarray.DataArray
            Input data.

        Returns
        -------
        xr.DataArray
            The inverse transformed components.

        """
        data = self.sanitizer.inverse_transform_components(data)
        data = self.postconverter.inverse_transform_components(data)
        data = self.stacker.inverse_transform_components(data)
        data = self.preconverter.inverse_transform_components(data)
        return self.scaler.inverse_transform_components(data)

    def inverse_transform_scores(self, data: DataArray) -> DataArray:
        """Inverse transform the scores.

        Parameters:
        -------------
        data: xr.DataArray or list of xarray.DataArray
            Input data.

        Returns
        -------
        xr.DataArray
            The inverse transformed scores.

        """
        data = self.sanitizer.inverse_transform_scores(data)
        data = self.postconverter.inverse_transform_scores(data)
        data = self.stacker.inverse_transform_scores(data)
        data = self.preconverter.inverse_transform_scores(data)
        return self.scaler.inverse_transform_scores(data)
