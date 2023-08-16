from typing import Optional, Sequence, Hashable, List

from .factory import StackerFactory, ScalerFactory, MultiIndexConverterFactory
from ..utils.xarray_utils import get_dims
from ..utils.data_types import AnyDataObject, DataArray


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

        # Define model parameters
        self._params = {
            "with_std": with_std,
            "with_coslat": with_coslat,
            "with_weights": with_weights,
        }

    def fit(
        self,
        data: AnyDataObject,
        dim: Hashable | Sequence[Hashable] | List[Sequence[Hashable]],
        weights: Optional[AnyDataObject] = None,
    ):
        """Just for consistency with the other classes."""
        raise NotImplementedError(
            "Preprocessor does not implement fit method. Use fit_transform instead."
        )

    def fit_transform(
        self,
        data: AnyDataObject,
        dim: Hashable | Sequence[Hashable] | List[Sequence[Hashable]],
        weights: Optional[AnyDataObject] = None,
    ) -> DataArray:
        """Preprocess the data.

        This will scale and stack the data.

        Parameters:
        -------------
        data: xr.DataArray or list of xarray.DataArray
            Input data.
        dim: tuple
            Tuple specifying the sample dimensions. The remaining dimensions
            will be treated as feature dimensions.
        weights: xr.DataArray or xr.Dataset or None, default=None
            If specified, the input data will be weighted by this array.

        """
        # Set sample and feature dimensions
        sample_dims, feature_dims = get_dims(data, sample_dims=dim)
        self.dims = {self.sample_name: sample_dims, self.feature_name: feature_dims}

        # Scale the data
        self.scaler = ScalerFactory.create_scaler(data, **self._params)
        data = self.scaler.fit_transform(data, sample_dims, feature_dims, weights)

        # Convert MultiIndex to single index
        self.converter = MultiIndexConverterFactory.create_converter(data)
        data = self.converter.fit_transform(data)  # type: ignore

        # Stack the data
        stacker_kwargs = {
            "sample_name": self.sample_name,
            "feature_name": self.feature_name,
        }
        self.stacker = StackerFactory.create_stacker(data, **stacker_kwargs)
        return self.stacker.fit_transform(data, sample_dims, feature_dims)

    def transform(self, data: AnyDataObject) -> DataArray:
        """Project new unseen data onto the components (EOFs/eigenvectors).

        Parameters:
        -------------
        data: xr.DataArray or list of xarray.DataArray
            Input data.

        Returns:
        ----------
        projections: DataArray | Dataset | List[DataArray]
            Projections of the new data onto the components.

        """
        data = self.scaler.transform(data)
        data = self.converter.transform(data)  # type: ignore
        return self.stacker.transform(data)

    def inverse_transform_data(self, data: DataArray) -> AnyDataObject:
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
        data = self.stacker.inverse_transform_data(data)
        data = self.converter.inverse_transform(data)  # type: ignore
        return self.scaler.inverse_transform(data)

    def inverse_transform_components(self, data: DataArray) -> AnyDataObject:
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
        data = self.stacker.inverse_transform_components(data)
        return self.converter.inverse_transform(data)  # type: ignore

    def inverse_transform_scores(self, data: DataArray) -> AnyDataObject:
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
        data = self.stacker.inverse_transform_scores(data)
        return self.converter.inverse_transform(data)  # type: ignore
