import pandas as pd
import xarray as xr
from typing_extensions import Self

from ..utils.data_types import Data, DataArray, DataSet, DataVar, DataVarBound, Dims
from .transformer import Transformer


class Stacker(Transformer):
    """Converts an xarray DataArray or Dataset of any dimensionality into a 2D DataArray.

    The new DataArray will have two dimensions: `sample` and `feature`.
    The dimensions of the original data will be stacked along these two dimensions.

    Attributes
    ----------
    sample_dims : Sequence[Hashable]
        The dimensions of the data that will be stacked along the `sample` dimension.
    feature_dims : Sequence[Hashable]
        The dimensions of the data that will be stacked along the `feature` dimension.
    sample_name : str
        The name of the sample dimension (dim=0).
    feature_name : str
        The name of the feature dimension (dim=1).
    dims_in : tuple[str]
        The dimensions of the input data.
    dims_out : tuple[str]
        The dimensions of the output data.
    dims_mapping : dict[str, tuple[str]]
        The mapping between the input and output dimensions.
    coords_in : dict[str, xr.Coordinates]
        The coordinates of the input data.
    coords_out : dict[str, xr.Coordinates]
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
        self.data_type = None

    def get_serialization_attrs(self) -> dict:
        return dict(
            dims_in=self.dims_in,
            dims_out=self.dims_out,
            dims_mapping=self.dims_mapping,
            coords_in=self.coords_in,
            coords_out=self.coords_out,
            data_type=self.data_type,
        )

    def _validate_data_type(self, X: Data):
        """Check that the data type is either DataArray or Dataset."""
        if not isinstance(X, (xr.DataArray, xr.Dataset)):
            raise TypeError(f"Invalid data type {type(X)}.")

    def _validate_dimension_names(self, X, sample_dims, feature_dims):
        """Check that the names of the sample and feature dimensions are not already present in the data."""
        sample_name_in_data = self.sample_name in X.dims
        feature_name_in_data = self.feature_name in X.dims

        has_invalid_sample_name = (
            True if (len(sample_dims) > 1) and sample_name_in_data else False
        )

        match X:
            case xr.DataArray():
                has_invalid_feature_name = (
                    True if (len(feature_dims) > 1) and feature_name_in_data else False
                )
            case xr.Dataset():
                has_invalid_feature_name = True if feature_name_in_data else False
            case _:
                raise TypeError(f"Invalid data type {type(X)}.")

        if has_invalid_sample_name:
            err_msg = f"Name of sample dimension ({self.sample_name}) is already present in data. Please use another name."
            raise ValueError(err_msg)

        if has_invalid_feature_name:
            err_msg = f"Name of feature dimension ({self.feature_name}) is already present in data. Please use another name."
            raise ValueError(err_msg)

    def _validate_dims(self, X: Data, sample_dims, feature_dims):
        invalid_sample_dims = True if len(sample_dims) < 1 else False
        invalid_feature_dims = True if len(feature_dims) < 1 else False

        if invalid_sample_dims:
            raise ValueError("Sample dimension must not be empty.")
        if invalid_feature_dims:
            match X:
                case xr.DataArray():
                    raise ValueError("Feature dimension must not be empty.")
                case xr.Dataset():
                    err_msg = "Dataset without feature dimension is currently not supported. Please convert your Dataset to a DataArray first, e.g. by using `to_array()`."
                    raise ValueError(err_msg)
                case _:
                    raise TypeError(f"Invalid data type {type(X)}.")

    def _validate_indices(self, X: Data):
        """Check that the indices of the data are no MultiIndex"""
        if any([isinstance(index, pd.MultiIndex) for index in X.indexes.values()]):
            raise ValueError("Cannot stack data containing a MultiIndex.")

    def _sanity_check(self, X: Data, sample_dims, feature_dims):
        self._validate_dims(X, sample_dims, feature_dims)
        self._validate_dimension_names(X, sample_dims, feature_dims)
        self._validate_indices(X)

    def _validate_transform_data_type(self, X: Data):
        """Check that the data type is either DataArray or Dataset."""
        if self._type_name(X) != self.data_type:
            raise TypeError(f"Expected data type {self.data_type}, got {type(X)}.")

    def _validate_transform_dimensions(self, X: Data):
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

    def _validate_transform_feature_coords(self, X: Data):
        """Verify that the feature coordinates of the data are consistent with the feature coordinates used to fit the stacker."""
        feature_dims = self.dims_mapping[self.feature_name]
        coords_are_equal = [
            X.coords[dim].equals(self.coords_in[dim]) for dim in feature_dims
        ]
        if not all(coords_are_equal):
            raise ValueError(
                "Data to be transformed has different coordinates than the data used to fit."
            )

    def _reorder_dims(self, X: DataVarBound) -> DataVarBound:
        """Reorder dimensions to original order; catch ('mode') dimensions via ellipsis"""
        order_input_dims = [
            valid_dim for valid_dim in self.dims_in if valid_dim in X.dims
        ]
        if order_input_dims != X.dims:
            X = X.transpose(..., *order_input_dims)
        return X

    def _stack(self, X: Data, sample_dims: Dims, feature_dims: Dims) -> DataArray:
        """Stack data to 2D.

        Parameters
        ----------
        X : xr.DataArray | xr.Dataset
            The data to be reshaped.
        sample_dims : Hashable or Sequence[Hashable]
            The dimensions of the data that will be stacked along the `sample` dimension.
        feature_dims : Hashable or Sequence[Hashable]
            The dimensions of the data that will be stacked along the `feature` dimension.

        Returns
        -------
        DataArray
            The reshaped 2d-data.
        """
        sample_name = self.sample_name
        feature_name = self.feature_name

        # Stack SAMPLE dimension
        if len(sample_dims) > 1:
            X = X.stack({sample_name: sample_dims})
        elif len(sample_dims) == 1:
            if sample_dims[0] != sample_name:
                X = X.rename({sample_dims[0]: sample_name})
            else:
                # There's only one sample dimension and it's already named correctly
                pass
        else:
            raise ValueError("Sample dimension must not be empty.")

        # Stack FEATURE dimension
        match X:
            case xr.DataArray():
                if len(feature_dims) > 1:
                    X = X.stack({feature_name: feature_dims})
                elif len(feature_dims) == 1:
                    if feature_dims[0] != feature_name:
                        X = X.rename({feature_dims[0]: feature_name})
                    else:
                        # There's only one feature dimension and it's already named correctly
                        pass
                else:
                    raise ValueError("Feature dimension must not be empty.")

            case xr.Dataset():
                X = X.to_stacked_array(
                    new_dim=feature_name, sample_dims=(self.sample_name,)
                )
            case _:
                raise TypeError(f"Invalid data type {type(X)}.")

        # Reorder dimensions to be always (sample, feature)
        if X.dims == (feature_name, sample_name):
            X = X.transpose(sample_name, feature_name)

        return X

    def _unstack_to_dataarray(self, X: DataArray) -> DataArray:
        """Unstack 2D DataArray to its original dimensions.

        Parameters
        ----------
        X : DataArray
            The data to be unstacked.

        Returns
        -------
        DataArray
            The unstacked data.
        """
        sample_name = self.sample_name
        feature_name = self.feature_name

        has_only_one_sample_dim = len(self.dims_mapping[sample_name]) == 1
        has_only_one_feature_dim = len(self.dims_mapping[feature_name]) == 1

        if sample_name in X.dims:
            # If sample dimensions is one dimensional, rename is sufficient, otherwise unstack
            if has_only_one_sample_dim:
                if self.dims_mapping[sample_name][0] != sample_name:
                    X = X.rename({sample_name: self.dims_mapping[sample_name][0]})
            else:
                X = X.unstack(sample_name)

        # pass if feature/sample dimensions do not exist in data
        if feature_name in X.dims:
            # If sample dimensions is one dimensional, rename is sufficient, otherwise unstack
            if has_only_one_feature_dim:
                if self.dims_mapping[feature_name][0] != feature_name:
                    X = X.rename({feature_name: self.dims_mapping[feature_name][0]})
            else:
                X = X.unstack(feature_name)

        else:
            pass

        X = self._reorder_dims(X)
        return X

    def _unstack_to_dataset_data(self, X: DataArray) -> DataSet:
        """Unstack `sample` and `feature` dimension of an DataArray to its original dimensions."""
        sample_name = self.sample_name
        feature_name = self.feature_name
        has_only_one_sample_dim = len(self.dims_mapping[sample_name]) == 1

        if has_only_one_sample_dim:
            X = X.rename({sample_name: self.dims_mapping[sample_name][0]})

        ds: DataSet = X.to_unstacked_dataset(feature_name, "variable").unstack()
        ds = self._reorder_dims(ds)
        return ds

    def _unstack_to_dataset_components(self, data: DataArray) -> DataSet:
        feature_name = self.feature_name
        ds: DataSet = data.to_unstacked_dataset(feature_name, "variable").unstack()
        ds = self._reorder_dims(ds)
        return ds

    def _type_name(self, X):
        """Store data type as a str so it is easily serializable."""
        return type(X).__name__

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
        self._sanity_check(X, sample_dims, feature_dims)

        self.data_type = self._type_name(X)
        self.sample_dims = sample_dims
        self.feature_dims = feature_dims
        self.dims_mapping.update(
            {
                self.sample_name: sample_dims,
                self.feature_name: feature_dims,
            }
        )

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
        self._validate_transform_dimensions(X)

        # Check if data to be transformed has the same feature coordinates as the data used to fit the stacker
        self._validate_transform_feature_coords(X)

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
        match self.data_type:
            case "DataArray":
                return self._unstack_to_dataarray(X)
            case "Dataset":
                return self._unstack_to_dataset_data(X)
            case _:
                raise TypeError(f"Invalid data type {self._type_name(X)}.")

    def inverse_transform_components(self, X: DataArray) -> Data:
        """Reshape the 2D components (feature x mode) back into its original dimensions.

        Parameters
        ----------
        X : DataArray
            The data to be reshaped.

        Returns
        -------
        DataArray
            The reshaped data.

        """
        match self.data_type:
            case "DataArray":
                return self._unstack_to_dataarray(X)
            case "Dataset":
                return self._unstack_to_dataset_components(X)
            case _:
                raise TypeError(f"Invalid data type {self._type_name(X)}.")

    def inverse_transform_scores(self, X: DataArray) -> DataArray:
        """Reshape the 2D scores (sample x mode) back into its original dimensions.

        Use this for fitted scores.

        Parameters
        ----------
        X : DataArray
            The data to be reshaped.

        Returns
        -------
        DataArray
            The reshaped data.

        """
        return self._unstack_to_dataarray(X)

    def inverse_transform_scores_unseen(self, X: DataArray) -> DataArray:
        """Reshape the 2D scores (sample x mode) back into its original dimensions.

        Use this for new, unseen scores.

        Parameters
        ----------
        X : DataArray
            The data to be reshaped.

        Returns
        -------
        DataArray
            The reshaped data.

        """
        return self.inverse_transform_scores(X)
