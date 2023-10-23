from abc import abstractmethod
from typing import List, Optional, Type
from typing_extensions import Self

import numpy as np
import pandas as pd
import xarray as xr

from .transformer import Transformer
from ..utils.data_types import Dims, DataArray, DataSet, Data, DataVar, DataVarBound
from ..utils.sanity_checks import convert_to_dim_type


class Stacker(Transformer):
    """Converts a DataArray of any dimensionality into a 2D structure.

    Attributes
    ----------
    sample_dims : Sequence[Hashable]
        The dimensions of the data that will be stacked along the `sample` dimension.
    feature_dims : Sequence[Hashable]
        The dimensions of the data that will be stacked along the `feature` dimension.
    sample_name : str
        The name of the sample dimension.
    feature_name : str
        The name of the feature dimension.
    dims_in : Tuple[str]
        The dimensions of the input data.
    dims_out : Tuple[str]
        The dimensions of the output data.
    dims_mapping : Dict[str, Tuple[str]]
        The mapping between the input and output dimensions.
    coords_in : Dict[str, xr.Coordinates]
        The coordinates of the input data.
    coords_out : Dict[str, xr.Coordinates]
        The coordinates of the output data.
    """

    def __init__(
        self,
        sample_name: str = "sample",
        feature_name: str = "feature",
    ):
        super().__init__(sample_name, feature_name)

        self.dims_in = tuple()
        self.dims_out = tuple((sample_name, feature_name))
        self.dims_mapping = {}
        self.dims_mapping.update({d: tuple() for d in self.dims_out})

        self.coords_in = {}
        self.coords_out = {}

    def _validate_matching_dimensions(self, X: Data):
        """Verify that the dimensions of the data are consistent with the dimensions used to fit the stacker."""
        # Test whether sample and feature dimensions are present in data array
        expected_sample_dims = set(self.dims_mapping[self.sample_name])
        expected_feature_dims = set(self.dims_mapping[self.feature_name])
        expected_dims = expected_sample_dims | expected_feature_dims
        given_dims = set(X.dims)
        if not (expected_dims == given_dims):
            raise ValueError(
                f"One or more dimensions in {expected_dims} are not present in data."
            )

    def _validate_matching_feature_coords(self, X: Data):
        """Verify that the feature coordinates of the data are consistent with the feature coordinates used to fit the stacker."""
        feature_dims = self.dims_mapping[self.feature_name]
        coords_are_equal = [
            X.coords[dim].equals(self.coords_in[dim]) for dim in feature_dims
        ]
        if not all(coords_are_equal):
            raise ValueError(
                "Data to be transformed has different coordinates than the data used to fit."
            )

    def _validate_dimension_names(self, sample_dims, feature_dims):
        if len(sample_dims) > 1:
            if self.sample_name in sample_dims:
                raise ValueError(
                    f"Name of sample dimension ({self.sample_name}) is already present in data. Please use another name."
                )
        if len(feature_dims) > 1:
            if self.feature_name in feature_dims:
                raise ValueError(
                    f"Name of feature dimension ({self.feature_name}) is already present in data. Please use another name."
                )

    def _validate_indices(self, X: Data):
        """Check that the indices of the data are no MultiIndex"""
        if any([isinstance(index, pd.MultiIndex) for index in X.indexes.values()]):
            raise ValueError(f"Cannot stack data containing a MultiIndex.")

    def _sanity_check(self, X: Data, sample_dims, feature_dims):
        self._validate_dimension_names(sample_dims, feature_dims)
        self._validate_indices(X)

    @abstractmethod
    def _stack(self, X: Data, sample_dims: Dims, feature_dims: Dims) -> DataArray:
        """Stack data to 2D.

        Parameters
        ----------
        data : DataArray
            The data to be reshaped.
        sample_dims : Hashable or Sequence[Hashable]
            The dimensions of the data that will be stacked along the `sample` dimension.
        feature_dims : Hashable or Sequence[Hashable]
            The dimensions of the data that will be stacked along the `feature` dimension.

        Returns
        -------
        data_stacked : DataArray
            The reshaped 2d-data.
        """

    @abstractmethod
    def _unstack(self, X: DataArray) -> Data:
        """Unstack 2D DataArray to its original dimensions.

        Parameters
        ----------
        data : DataArray
            The data to be unstacked.

        Returns
        -------
        data_unstacked : DataArray
            The unstacked data.
        """

    def _reorder_dims(self, X: DataVarBound) -> DataVarBound:
        """Reorder dimensions to original order; catch ('mode') dimensions via ellipsis"""
        order_input_dims = [
            valid_dim for valid_dim in self.dims_in if valid_dim in X.dims
        ]
        if order_input_dims != X.dims:
            X = X.transpose(..., *order_input_dims)
        return X

    def fit(self, X: Data, sample_dims: Dims, feature_dims: Dims) -> Self:
        """Fit the stacker.

        Parameters
        ----------
        data : DataArray
            The data to be reshaped.

        Returns
        -------
        self : DataArrayStacker
            The fitted stacker.

        """
        self.sample_dims = sample_dims
        self.feature_dims = feature_dims
        self.dims_mapping.update(
            {
                self.sample_name: sample_dims,
                self.feature_name: feature_dims,
            }
        )
        self._sanity_check(X, sample_dims, feature_dims)

        # Set dimensions and coordinates
        self.dims_in = X.dims
        self.coords_in = {dim: X.coords[dim] for dim in X.dims}

        return self

    def transform(self, X: Data) -> DataArray:
        """Reshape DataArray to 2D.

        Parameters
        ----------
        X : DataArray
            The data to be reshaped.

        Returns
        -------
        DataArray
            The reshaped data.

        Raises
        ------
        ValueError
            If the data to be transformed has different dimensions than the data used to fit the stacker.
        ValueError
            If the data to be transformed has different coordinates than the data used to fit the stacker.

        """
        # Test whether sample and feature dimensions are present in data array
        self._validate_matching_dimensions(X)

        # Check if data to be transformed has the same feature coordinates as the data used to fit the stacker
        self._validate_matching_feature_coords(X)

        # Stack data
        sample_dims = self.dims_mapping[self.sample_name]
        feature_dims = self.dims_mapping[self.feature_name]
        da: DataArray = self._stack(
            X, sample_dims=sample_dims, feature_dims=feature_dims
        )

        # Set out coordinates
        self.coords_out.update(
            {
                self.sample_name: da.coords[self.sample_name],
                self.feature_name: da.coords[self.feature_name],
            }
        )
        return da

    def fit_transform(
        self,
        X: DataVar,
        sample_dims: Dims,
        feature_dims: Dims,
    ) -> DataArray:
        return self.fit(X, sample_dims, feature_dims).transform(X)

    def inverse_transform_data(self, X: DataArray) -> Data:
        """Reshape the 2D data (sample x feature) back into its original dimensions.

        Parameters
        ----------
        X : DataArray
            The data to be reshaped.

        Returns
        -------
        DataArray
            The reshaped data.

        """
        Xnd = self._unstack(X)
        Xnd = self._reorder_dims(Xnd)
        return Xnd

    def inverse_transform_components(self, X: DataArray) -> Data:
        """Reshape the 2D components (sample x feature) back into its original dimensions.

        Parameters
        ----------
        data : DataArray
            The data to be reshaped.

        Returns
        -------
        DataArray
            The reshaped data.

        """
        Xnd = self._unstack(X)
        Xnd = self._reorder_dims(Xnd)
        return Xnd

    def inverse_transform_scores(self, data: DataArray) -> DataArray:
        """Reshape the 2D scores (sample x feature) back into its original dimensions.

        Parameters
        ----------
        data : DataArray
            The data to be reshaped.

        Returns
        -------
        DataArray
            The reshaped data.

        """
        data = self._unstack(data)  # type: ignore
        data = self._reorder_dims(data)
        return data


class DataArrayStacker(Stacker):
    def _stack(
        self, data: DataArray, sample_dims: Dims, feature_dims: Dims
    ) -> DataArray:
        """Reshape a DataArray to 2D.

        Parameters
        ----------
        data : DataArray
            The data to be reshaped.
        sample_dims : Hashable or Sequence[Hashable]
            The dimensions of the data that will be stacked along the `sample` dimension.
        feature_dims : Hashable or Sequence[Hashable]
            The dimensions of the data that will be stacked along the `feature` dimension.

        Returns
        -------
        data_stacked : DataArray
            The reshaped 2d-data.
        """
        sample_name = self.sample_name
        feature_name = self.feature_name

        # 3 cases:
        # 1. uni-dimensional with correct feature/sample name ==> do nothing
        # 2. uni-dimensional with name different from feature/sample ==> rename
        # 3. multi-dimensinoal with names different from feature/sample ==> stack

        # - SAMPLE -
        if len(sample_dims) == 1:
            # Case 1
            if sample_dims[0] == sample_name:
                pass
            # Case 2
            else:
                data = data.rename({sample_dims[0]: sample_name})
        # Case 3
        else:
            data = data.stack({sample_name: sample_dims})

        # - FEATURE -
        if len(feature_dims) == 1:
            # Case 1
            if feature_dims[0] == feature_name:
                pass
            # Case 2
            else:
                data = data.rename({feature_dims[0]: feature_name})
        # Case 3
        else:
            data = data.stack({feature_name: feature_dims})

        # Reorder dimensions to be always (sample, feature)
        if data.dims == (feature_name, sample_name):
            data = data.transpose(sample_name, feature_name)

        return data

    def _unstack(self, data: DataArray) -> DataArray:
        """Unstack 2D DataArray to its original dimensions.

        Parameters
        ----------
        data : DataArray
            The data to be unstacked.

        Returns
        -------
        data_unstacked : DataArray
            The unstacked data.
        """
        sample_name = self.sample_name
        feature_name = self.feature_name

        # pass if feature/sample dimensions do not exist in data
        if feature_name in data.dims:
            # If sample dimensions is one dimensional, rename is sufficient, otherwise unstack
            if len(self.dims_mapping[feature_name]) == 1:
                if self.dims_mapping[feature_name][0] != feature_name:
                    data = data.rename(
                        {feature_name: self.dims_mapping[feature_name][0]}
                    )
            else:
                data = data.unstack(feature_name)

        if sample_name in data.dims:
            # If sample dimensions is one dimensional, rename is sufficient, otherwise unstack
            if len(self.dims_mapping[sample_name]) == 1:
                if self.dims_mapping[sample_name][0] != sample_name:
                    data = data.rename({sample_name: self.dims_mapping[sample_name][0]})
            else:
                data = data.unstack(sample_name)

        else:
            pass

        return data


class DataSetStacker(Stacker):
    """Converts a Dataset of any dimensionality into a 2D structure."""

    def _validate_dimension_names(self, sample_dims, feature_dims):
        if len(sample_dims) > 1:
            if self.sample_name in sample_dims:
                raise ValueError(
                    f"Name of sample dimension ({self.sample_name}) is already present in data. Please use another name."
                )
        if len(feature_dims) >= 1:
            if self.feature_name in feature_dims:
                raise ValueError(
                    f"Name of feature dimension ({self.feature_name}) is already present in data. Please use another name."
                )
        else:
            raise ValueError(
                f"Datasets without feature dimension are currently not supported. Please convert your Dataset to a DataArray first, e.g. by using `to_array()`."
            )

    def _stack(self, data: DataSet, sample_dims, feature_dims) -> DataArray:
        """Reshape a Dataset to 2D.

        Parameters
        ----------
        data : Dataset
            The data to be reshaped.
        sample_dims : Hashable or Sequence[Hashable]
            The dimensions of the data that will be stacked along the `sample` dimension.
        feature_dims : Hashable or Sequence[Hashable]
            The dimensions of the data that will be stacked along the `feature` dimension.

        Returns
        -------
        data_stacked : DataArray
            The reshaped 2d-data.
        """
        sample_name = self.sample_name
        feature_name = self.feature_name

        # 3 cases:
        # 1. uni-dimensional with correct feature/sample name ==> do nothing
        # 2. uni-dimensional with name different from feature/sample ==> rename
        # 3. multi-dimensinoal with names different from feature/sample ==> stack

        # - SAMPLE -
        if len(sample_dims) == 1:
            # Case 1
            if sample_dims[0] == sample_name:
                pass
            # Case 2
            else:
                data = data.rename({sample_dims[0]: sample_name})
        # Case 3
        else:
            data = data.stack({sample_name: sample_dims})

        # - FEATURE -
        # Convert Dataset -> DataArray, stacking all non-sample dimensions to feature dimension, including data variables
        err_msg = f"Feature dimension {feature_dims[0]} already exists in data. Please choose another feature dimension name."
        # Case 2 & 3
        if (len(feature_dims) == 1) & (feature_dims[0] == feature_name):
            raise ValueError(err_msg)
        else:
            try:
                da = data.to_stacked_array(
                    new_dim=feature_name, sample_dims=(self.sample_name,)
                )
            except ValueError:
                raise ValueError(err_msg)

        # Reorder dimensions to be always (sample, feature)
        if da.dims == (feature_name, sample_name):
            da = da.transpose(sample_name, feature_name)

        return da

    def _unstack_data(self, data: DataArray) -> DataSet:
        """Unstack `sample` and `feature` dimension of an DataArray to its original dimensions."""
        sample_name = self.sample_name
        feature_name = self.feature_name
        has_only_one_sample_dim = len(self.dims_mapping[sample_name]) == 1

        if has_only_one_sample_dim:
            data = data.rename({sample_name: self.dims_mapping[sample_name][0]})

        ds: DataSet = data.to_unstacked_dataset(feature_name, "variable").unstack()
        ds = self._reorder_dims(ds)
        return ds

    def _unstack_components(self, data: DataArray) -> DataSet:
        feature_name = self.feature_name
        ds: DataSet = data.to_unstacked_dataset(feature_name, "variable").unstack()
        ds = self._reorder_dims(ds)
        return ds

    def _unstack_scores(self, data: DataArray) -> DataArray:
        sample_name = self.sample_name
        has_only_one_sample_dim = len(self.dims_mapping[sample_name]) == 1

        if has_only_one_sample_dim:
            data = data.rename({sample_name: self.dims_mapping[sample_name][0]})

        data = data.unstack()
        data = self._reorder_dims(data)
        return data

    def inverse_transform_data(self, X: DataArray) -> DataSet:
        """Reshape the 2D data (sample x feature) back into its original shape."""
        X_ds: DataSet = self._unstack_data(X)
        return X_ds

    def inverse_transform_components(self, X: DataArray) -> DataSet:
        """Reshape the 2D components (sample x feature) back into its original shape."""
        X_ds: DataSet = self._unstack_components(X)
        return X_ds

    def inverse_transform_scores(self, X: DataArray) -> DataArray:
        """Reshape the 2D scores (sample x feature) back into its original shape."""
        X = self._unstack_scores(X)
        return X


class StackerFactory:
    """Factory class for creating stackers."""

    def __init__(self):
        pass

    @staticmethod
    def create(data: Data) -> Type[DataArrayStacker] | Type[DataSetStacker]:
        """Create a stacker for the given data."""
        if isinstance(data, xr.DataArray):
            return DataArrayStacker
        elif isinstance(data, xr.Dataset):
            return DataSetStacker
        else:
            raise TypeError(f"Invalid data type {type(data)}.")
