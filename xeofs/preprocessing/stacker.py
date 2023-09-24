from typing import List, Self

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin

from xeofs.utils.data_types import DataArray, DataSet, DataList

from ..utils.data_types import (
    Dims,
    DimsList,
    DataArray,
    Dataset,
)
from ..utils.sanity_checks import convert_to_dim_type


class DataArrayStacker(BaseEstimator, TransformerMixin):
    """Converts a DataArray of any dimensionality into a 2D structure.

    Attributes
    ----------
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
        self.sample_name = sample_name
        self.feature_name = feature_name

        self.dims_in = tuple()
        self.dims_out = tuple((sample_name, feature_name))
        self.dims_mapping = {d: tuple() for d in self.dims_out}

        self.coords_in = {}
        self.coords_out = {}

    def _validate_matching_dimensions(self, data: DataArray):
        """Verify that the dimensions of the data are consistent with the dimensions used to fit the stacker."""
        # Test whether sample and feature dimensions are present in data array
        expected_sample_dims = set(self.dims_mapping[self.sample_name])
        expected_feature_dims = set(self.dims_mapping[self.feature_name])
        expected_dims = expected_sample_dims | expected_feature_dims
        given_dims = set(data.dims)
        if not (expected_dims == given_dims):
            raise ValueError(
                f"One or more dimensions in {expected_dims} are not present in data."
            )

    def _validate_matching_feature_coords(self, data: DataArray):
        """Verify that the feature coordinates of the data are consistent with the feature coordinates used to fit the stacker."""
        feature_dims = self.dims_mapping[self.feature_name]
        coords_are_equal = [
            data.coords[dim].equals(self.coords_in[dim]) for dim in feature_dims
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

    def _validate_indices(self, data: DataArray):
        """Check that the indices of the data are no MultiIndex"""
        if any([isinstance(index, pd.MultiIndex) for index in data.indexes.values()]):
            raise ValueError(f"Cannot stack data containing a MultiIndex.")

    def _sanity_check(self, data: DataArray, sample_dims, feature_dims):
        self._validate_dimension_names(sample_dims, feature_dims)
        self._validate_indices(data)

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

    def _reorder_dims(self, data):
        """Reorder dimensions to original order; catch ('mode') dimensions via ellipsis"""
        order_input_dims = [
            valid_dim for valid_dim in self.dims_in if valid_dim in data.dims
        ]
        if order_input_dims != data.dims:
            data = data.transpose(..., *order_input_dims)
        return data

    def fit(
        self,
        data: DataArray,
        sample_dims: Dims,
        feature_dims: Dims,
        y=None,
    ) -> Self:
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
        self._sanity_check(data, sample_dims, feature_dims)

        # Set in/out dimensions
        self.dims_in = data.dims
        self.dims_mapping = {
            self.sample_name: sample_dims,
            self.feature_name: feature_dims,
        }

        # Set in coordinates
        self.coords_in = {dim: data.coords[dim] for dim in data.dims}

        return self

    def transform(self, data: DataArray) -> DataArray:
        """Reshape DataArray to 2D.

        Parameters
        ----------
        data : DataArray
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
        self._validate_matching_dimensions(data)

        # Check if data to be transformed has the same feature coordinates as the data used to fit the stacker
        self._validate_matching_feature_coords(data)

        # Stack data
        sample_dims = self.dims_mapping[self.sample_name]
        feature_dims = self.dims_mapping[self.feature_name]
        da: DataArray = self._stack(
            data, sample_dims=sample_dims, feature_dims=feature_dims
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
        data: DataArray,
        sample_dims: Dims,
        feature_dims: Dims,
        y=None,
    ) -> DataArray:
        return self.fit(data, sample_dims, feature_dims, y).transform(data)

    def inverse_transform_data(self, data: DataArray) -> DataArray:
        """Reshape the 2D data (sample x feature) back into its original dimensions.

        Parameters
        ----------
        data : DataArray
            The data to be reshaped.

        Returns
        -------
        DataArray
            The reshaped data.

        """
        data = self._unstack(data)
        data = self._reorder_dims(data)
        return data

    def inverse_transform_components(self, data: DataArray) -> DataArray:
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
        data = self._unstack(data)
        data = self._reorder_dims(data)
        return data

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
        data = self._unstack(data)
        data = self._reorder_dims(data)
        return data


class DataSetStacker(DataArrayStacker):
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

    def _stack(self, data: Dataset, sample_dims, feature_dims) -> DataArray:
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

    def fit(
        self,
        data: DataSet,
        sample_dims: Dims,
        feature_dims: Dims,
        y=None,
    ) -> Self:
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
        return super().fit(data, sample_dims, feature_dims, y)  # type: ignore

    def transform(self, data: DataSet) -> DataArray:
        return super().transform(data)  # type: ignore

    def fit_transform(
        self, data: DataSet, sample_dims: Dims, feature_dims: Dims, y=None
    ) -> DataArray:
        return super().fit_transform(data, sample_dims, feature_dims, y)  # type: ignore

    def inverse_transform_data(self, data: DataArray) -> DataSet:
        """Reshape the 2D data (sample x feature) back into its original shape."""
        data_ds: DataSet = self._unstack_data(data)
        return data_ds

    def inverse_transform_components(self, data: DataArray) -> DataSet:
        """Reshape the 2D components (sample x feature) back into its original shape."""
        data_ds: DataSet = self._unstack_components(data)
        return data_ds

    def inverse_transform_scores(self, data: DataArray) -> DataArray:
        """Reshape the 2D scores (sample x feature) back into its original shape."""
        data = self._unstack_scores(data)
        return data


class DataListStacker(DataArrayStacker):
    """Converts a list of DataArrays of any dimensionality into a 2D structure.

    This operation generates a reshaped DataArray with two distinct dimensions: 'sample' and 'feature'.

    At a minimum, the `sample` dimension must be present in all DataArrays. The `feature` dimension can be different
    for each DataArray and must be specified as a list of dimensions.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stackers = []

    def fit(
        self,
        X: DataList,
        sample_dims: Dims,
        feature_dims: DimsList,
        y=None,
    ):
        """Fit the stacker.

        Parameters
        ----------
        X : DataArray
            The data to be reshaped.

        Returns
        -------
        self : DataArrayStacker
            The fitted stacker.

        """

        # Check input
        if not isinstance(feature_dims, list):
            raise TypeError(
                "feature dims must be a list of the feature dimensions of each DataArray"
            )

        sample_dims = convert_to_dim_type(sample_dims)
        feature_dims = [convert_to_dim_type(fdims) for fdims in feature_dims]

        if len(X) != len(feature_dims):
            err_message = (
                "Number of data arrays and feature dimensions must be the same. "
            )
            err_message += (
                f"Got {len(X)} data arrays and {len(feature_dims)} feature dimensions"
            )
            raise ValueError(err_message)

        # Set in/out dimensions
        self.dims_in = [data.dims for data in X]
        self.dims_out = tuple((self.sample_name, self.feature_name))
        self.dims_mapping = {
            self.sample_name: sample_dims,
            self.feature_name: feature_dims,
        }

        # Set in/out coordinates
        self.coords_in = [{dim: data.coords[dim] for dim in data.dims} for data in X]

        # Fit stacker for each DataArray
        for data, fdims in zip(X, feature_dims):
            stacker = DataArrayStacker(
                sample_name=self.sample_name, feature_name=self.feature_name
            )
            stacker.fit(data, sample_dims=sample_dims, feature_dims=fdims)
            self.stackers.append(stacker)

        return self

    def transform(self, X: DataList) -> DataArray:
        """Reshape DataArray to 2D.

        Parameters
        ----------
        X : DataList
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
        # Test whether the input list has same length as the number of stackers
        if len(X) != len(self.stackers):
            raise ValueError(
                f"Invalid input. Number of DataArrays ({len(X)}) does not match the number of fitted DataArrays ({len(self.stackers)})."
            )

        stacked_data_list: List[DataArray] = []
        idx_coords_size = []
        dummy_feature_coords = []

        # Stack individual DataArrays
        for stacker, data in zip(self.stackers, X):
            data_stacked = stacker.transform(data)
            idx_coords_size.append(data_stacked.coords[self.feature_name].size)
            stacked_data_list.append(data_stacked)

        # Create dummy feature coordinates for each DataArray
        idx_range = np.cumsum([0] + idx_coords_size)
        for i in range(len(idx_range) - 1):
            dummy_feature_coords.append(np.arange(idx_range[i], idx_range[i + 1]))

        # Replace original feature coordiantes with dummy coordinates
        for i, data in enumerate(stacked_data_list):
            data = data.drop_vars(self.feature_name)
            stacked_data_list[i] = data.assign_coords(
                {self.feature_name: dummy_feature_coords[i]}
            )

        self._dummy_feature_coords = dummy_feature_coords

        stacked_data: DataArray = xr.concat(stacked_data_list, dim=self.feature_name)

        self.coords_out = {
            self.sample_name: stacked_data.coords[self.sample_name],
            self.feature_name: stacked_data.coords[self.feature_name],
        }
        return stacked_data

    def fit_transform(
        self,
        X: DataList,
        sample_dims: Dims,
        feature_dims: DimsList,
        y=None,
    ) -> DataArray:
        return self.fit(X, sample_dims, feature_dims, y).transform(X)

    def _split_dataarray_into_list(self, data: DataArray) -> DataList:
        feature_name = self.feature_name
        data_list: DataList = []

        for stacker, features in zip(self.stackers, self._dummy_feature_coords):
            # Select the features corresponding to the current DataArray
            sub_selection = data.sel({feature_name: features})
            # Replace dummy feature coordinates with original feature coordinates
            sub_selection = sub_selection.assign_coords(
                {feature_name: stacker.coords_out[feature_name]}
            )

            # In case of MultiIndex we have to set the index to the feature dimension again
            if isinstance(sub_selection.indexes[feature_name], pd.MultiIndex):
                sub_selection = sub_selection.set_index(
                    {feature_name: stacker.dims_mapping[feature_name]}
                )
            else:
                # NOTE: This is a workaround for the case where the feature dimension is a tuple of length 1
                # the problem is described here: https://github.com/pydata/xarray/discussions/7958
                sub_selection = sub_selection.rename(
                    {feature_name: stacker.dims_mapping[feature_name][0]}
                )
            data_list.append(sub_selection)

        return data_list

    def inverse_transform_data(self, data: DataArray) -> DataList:
        """Reshape the 2D data (sample x feature) back into its original shape."""
        data_split: DataList = self._split_dataarray_into_list(data)
        data_transformed = []
        for stacker, data in zip(self.stackers, data_split):
            # Inverse transform the data using the corresponding stacker
            data_transformed.append(stacker.inverse_transform_data(data))

        return data_transformed

    def inverse_transform_components(self, data: DataArray) -> DataList:
        """Reshape the 2D components (sample x feature) back into its original shape."""
        data_split: DataList = self._split_dataarray_into_list(data)

        data_transformed = []
        for stacker, data in zip(self.stackers, data_split):
            # Inverse transform the data using the corresponding stacker
            data_transformed.append(stacker.inverse_transform_components(data))

        return data_transformed

    def inverse_transform_scores(self, data: DataArray) -> DataArray:
        """Reshape the 2D scores (sample x mode) back into its original shape."""
        return self.stackers[0].inverse_transform_scores(data)
