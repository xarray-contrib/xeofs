import numpy as np
import pandas as pd
from xeofs.utils.data_types import (
    DataArray,
    DataSet,
    DataList,
    Dims,
    DimsList,
    DimsTuple,
    DimsListTuple,
)
from xeofs.utils.xarray_utils import data_is_dask  # noqa: F401


def is_xdata(data):
    return isinstance(data, (DataArray, DataSet))


def get_dims_from_data(data: DataArray | DataSet) -> DimsTuple:
    # If data is DataArray/Dataset
    if is_xdata(data):
        data_dims: Dims = tuple(data.dims)
        sample_dims: Dims = tuple([dim for dim in data.dims if "sample" in str(dim)])
        feature_dims: Dims = tuple([dim for dim in data.dims if "feature" in str(dim)])
        return data_dims, sample_dims, feature_dims
    else:
        raise ValueError("unrecognized input type")


def get_dims_from_data_list(data_list: DataList) -> DimsListTuple:
    # If data is list
    if isinstance(data_list, list):
        data_dims: DimsList = [data.dims for data in data_list]
        sample_dims: DimsList = []
        feature_dims: DimsList = []
        for data in data_list:
            sdims = tuple([dim for dim in data.dims if "sample" in str(dim)])
            fdims = tuple([dim for dim in data.dims if "feature" in str(dim)])
            sample_dims.append(sdims)
            feature_dims.append(fdims)
        return data_dims, sample_dims, feature_dims

    else:
        raise ValueError("unrecognized input type")


def data_has_multiindex(data: DataArray | DataSet | DataList) -> bool:
    """Check if the given data object has any MultiIndex."""
    if isinstance(data, DataArray) or isinstance(data, DataSet):
        return any(isinstance(index, pd.MultiIndex) for index in data.indexes.values())
    elif isinstance(data, list):
        return all(data_has_multiindex(da) for da in data)
    else:
        raise ValueError("unrecognized input type")


def assert_expected_dims(data1, data2, policy="all"):
    """
    Check if dimensions of two data objects matches.

    Parameters:
    - data1: Reference data object (either a DataArray, DataSet, or list of DataArray)
    - data2: Test data object (same type as data1)
    - policy: Policy to check the dimensions. Can be either "all", "feature" or "sample"

    """

    if is_xdata(data1) and is_xdata(data2):
        all_dims1, sample_dims1, feature_dims1 = get_dims_from_data(data1)
        all_dims2, sample_dims2, feature_dims2 = get_dims_from_data(data2)

        if policy == "all":
            err_msg = "Dimensions do not match: {:} vs {:}".format(all_dims1, all_dims2)
            assert set(all_dims1) == set(all_dims2), err_msg
        elif policy == "feature":
            err_msg = "Dimensions do not match: {:} vs {:}".format(
                feature_dims1, feature_dims2
            )
            assert set(feature_dims1) == set(feature_dims2), err_msg
            assert len(sample_dims2) == 0, "Sample dimensions should be empty"
            assert "mode" in all_dims2, "Mode dimension is missing"

        elif policy == "sample":
            err_msg = "Dimensions do not match: {:} vs {:}".format(
                sample_dims1, sample_dims2
            )
            assert set(sample_dims1) == set(sample_dims2), err_msg
            assert len(feature_dims2) == 0, "Feature dimensions should be empty"
            assert "mode" in all_dims2, "Mode dimension is missing"
        else:
            raise ValueError("Unrecognized policy: {:}".format(policy))

    elif isinstance(data1, list) and isinstance(data2, list):
        for da1, da2 in zip(data1, data2):
            assert_expected_dims(da1, da2, policy=policy)

    # If neither of the above conditions are met, raise an error
    else:
        raise ValueError(
            "Cannot check coordinates. Unrecognized data type. data1: {:}, data2: {:}".format(
                type(data1), type(data2)
            )
        )


def assert_expected_coords(data1, data2, policy="all") -> None:
    """
    Check if coordinates of the data objects matches.

    Parameters:
    - data1: Reference data object (either a DataArray, DataSet, or list of DataArray)
    - data2: Test data object (same type as data1)
    - policy: Policy to check the dimensions. Can be either "all", "feature" or "sample"

    """

    # Data objects is either DataArray or DataSet
    if is_xdata(data1) and is_xdata(data2):
        all_dims1, sample_dims1, feature_dims1 = get_dims_from_data(data1)
        all_dims2, sample_dims2, feature_dims2 = get_dims_from_data(data2)
        if policy == "all":
            assert all(
                np.all(data1.coords[dim].values == data2.coords[dim].values)
                for dim in all_dims1
            )
        elif policy == "feature":
            assert all(
                np.all(data1.coords[dim].values == data2.coords[dim].values)
                for dim in feature_dims1
            )
        elif policy == "sample":
            assert all(
                np.all(data1.coords[dim].values == data2.coords[dim].values)
                for dim in sample_dims1
            )
        else:
            raise ValueError("Unrecognized policy: {:}".format(policy))

    # Data object is list
    elif isinstance(data1, list) and isinstance(data2, list):
        for da1, da2 in zip(data1, data2):
            assert_expected_coords(da1, da2, policy=policy)

    # If neither of the above conditions are met, raise an error
    else:
        raise ValueError(
            "Cannot check coordinates. Unrecognized data type. data1: {:}, data2: {:}".format(
                type(data1), type(data2)
            )
        )
