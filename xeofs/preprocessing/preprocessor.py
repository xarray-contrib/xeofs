from typing import Optional, Sequence, Hashable, List, Tuple, Any, Type

import numpy as np

from .list_processor import GenericListTransformer
from .dimension_renamer import DimensionRenamer
from .scaler import Scaler
from .stacker import StackerFactory, Stacker
from .multi_index_converter import MultiIndexConverter
from .sanitizer import Sanitizer
from .concatenator import Concatenator
from ..utils.xarray_utils import (
    get_dims,
    unwrap_singleton_list,
    process_parameter,
    _check_parameter_number,
    convert_to_list,
)
from ..utils.data_types import (
    DataArray,
    Data,
    DataVar,
    DataVarBound,
    DataList,
    Dims,
    DimsList,
)


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
    new_sample_dims: Dims = tuple(np.unique(np.asarray(new_sample_dims)))
    return new_sample_dims, new_feature_dims


class Preprocessor:
    """Scale and stack the data along sample dimensions.

    Scaling includes (i) removing the mean and, optionally, (ii) dividing by the standard deviation,
    (iii) multiplying by the square root of cosine of latitude weights (area weighting; coslat weighting),
    and (iv) multiplying by additional user-defined weights.

    Stacking includes (i) stacking the data along the sample dimensions and (ii) stacking the data along the feature dimensions.

    Parameters
    ----------
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
        If True, the output is returned as a list of DataArrays. If False, the output is returned as a single DataArray if possible.

    """

    def __init__(
        self,
        sample_name: str = "sample",
        feature_name: str = "feature",
        with_center: bool = True,
        with_std: bool = False,
        with_coslat: bool = False,
        return_list: bool = True,
    ):
        # Set parameters
        self.sample_name = sample_name
        self.feature_name = feature_name
        self.with_center = with_center
        self.with_std = with_std
        self.with_coslat = with_coslat
        self.return_list = return_list

    def fit(
        self,
        X: List[Data] | Data,
        sample_dims: Dims,
        weights: Optional[List[Data] | Data] = None,
    ):
        self._set_return_list(X)
        X = convert_to_list(X)
        self.n_data = len(X)
        sample_dims, feature_dims = get_dims(X, sample_dims)

        # Set sample and feature dimensions
        self.dims = {
            self.sample_name: sample_dims,
            self.feature_name: feature_dims,
        }

        # However, for each DataArray a list of feature dimensions must be provided
        _check_parameter_number("feature_dims", feature_dims, self.n_data)

        # Ensure that weights are provided as a list
        weights = process_parameter("weights", weights, None, self.n_data)

        # 1 | Center, scale and weigh the data
        scaler_kwargs = {
            "with_center": self.with_center,
            "with_std": self.with_std,
            "with_coslat": self.with_coslat,
        }
        scaler_ikwargs = {
            "weights": weights,
        }
        self.scaler = GenericListTransformer(Scaler, **scaler_kwargs)
        X = self.scaler.fit_transform(X, sample_dims, feature_dims, scaler_ikwargs)

        # 2 | Rename dimensions
        self.renamer = GenericListTransformer(DimensionRenamer)
        X = self.renamer.fit_transform(X, sample_dims, feature_dims)
        sample_dims, feature_dims = extract_new_dim_names(self.renamer.transformers)

        # 3 | Convert MultiIndexes (before stacking)
        self.preconverter = GenericListTransformer(MultiIndexConverter)
        X = self.preconverter.fit_transform(X, sample_dims, feature_dims)

        # 4 | Stack the data to 2D DataArray
        stacker_kwargs = {
            "sample_name": self.sample_name,
            "feature_name": self.feature_name,
        }
        stack_type: Type[Stacker] = StackerFactory.create(X[0])
        self.stacker = GenericListTransformer(stack_type, **stacker_kwargs)
        X = self.stacker.fit_transform(X, sample_dims, feature_dims)
        # 5 | Convert MultiIndexes (after stacking)
        self.postconverter = GenericListTransformer(MultiIndexConverter)
        X = self.postconverter.fit_transform(X, sample_dims, feature_dims)
        # 6 | Remove NaNs
        sanitizer_kwargs = {
            "sample_name": self.sample_name,
            "feature_name": self.feature_name,
        }
        self.sanitizer = GenericListTransformer(Sanitizer, **sanitizer_kwargs)
        X = self.sanitizer.fit_transform(X, sample_dims, feature_dims)

        # 7 | Concatenate into one 2D DataArray
        self.concatenator = Concatenator(self.sample_name, self.feature_name)
        self.concatenator.fit(X)  # type: ignore

        return self

    def transform(self, X: List[Data] | Data) -> DataArray:
        X = convert_to_list(X)

        if len(X) != self.n_data:
            raise ValueError(
                f"number of data objects passed should match number of data objects used for fitting"
                f"len(data objects)={len(X)} and "
                f"len(data objects used for fitting)={self.n_data}"
            )

        X = self.scaler.transform(X)
        X = self.renamer.transform(X)
        X = self.preconverter.transform(X)
        X = self.stacker.transform(X)
        X = self.postconverter.transform(X)
        X = self.sanitizer.transform(X)
        return self.concatenator.transform(X)  # type: ignore

    def fit_transform(
        self,
        X: List[Data] | Data,
        sample_dims: Dims,
        weights: Optional[List[Data] | Data] = None,
    ) -> DataArray:
        return self.fit(X, sample_dims, weights).transform(X)

    def inverse_transform_data(self, X: DataArray) -> List[Data] | Data:
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
        X_list = self.concatenator.inverse_transform_data(X)
        X_list = self.sanitizer.inverse_transform_data(X_list)  # type: ignore
        X_list = self.postconverter.inverse_transform_data(X_list)
        X_list_ND = self.stacker.inverse_transform_data(X_list)
        X_list_ND = self.preconverter.inverse_transform_data(X_list_ND)
        X_list_ND = self.renamer.inverse_transform_data(X_list_ND)
        X_list_ND = self.scaler.inverse_transform_data(X_list_ND)
        return self._process_output(X_list_ND)

    def inverse_transform_components(self, X: DataArray) -> List[Data] | Data:
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
        X_list = self.concatenator.inverse_transform_components(X)
        X_list = self.sanitizer.inverse_transform_components(X_list)  # type: ignore
        X_list = self.postconverter.inverse_transform_components(X_list)
        X_list_ND = self.stacker.inverse_transform_components(X_list)
        X_list_ND = self.preconverter.inverse_transform_components(X_list_ND)
        X_list_ND = self.renamer.inverse_transform_components(X_list_ND)
        X_list_ND = self.scaler.inverse_transform_components(X_list_ND)
        return self._process_output(X_list_ND)

    def inverse_transform_scores(self, X: DataArray) -> DataArray:
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
        X_list = self.concatenator.inverse_transform_scores(X)
        X_list = self.sanitizer.inverse_transform_scores(X_list)
        X_list = self.postconverter.inverse_transform_scores(X_list)
        X_list_ND = self.stacker.inverse_transform_scores(X_list)
        X_list_ND = self.preconverter.inverse_transform_scores(X_list_ND)
        X_list_ND = self.renamer.inverse_transform_scores(X_list_ND)
        X_list_ND = self.scaler.inverse_transform_scores(X_list_ND)
        return X_list_ND

    def _process_output(self, X: List[Data]) -> List[Data] | Data:
        if self.return_list:
            return X
        else:
            return unwrap_singleton_list(X)

    def _set_return_list(self, X):
        if isinstance(X, (list, tuple)):
            self.return_list = True
        else:
            self.return_list = False
