from abc import ABC
from typing import Optional, Dict
from typing_extensions import Self
from abc import abstractmethod

import pandas as pd
import xarray as xr
from datatree import DataTree
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils.data_types import Dims, DataVar, DataArray, DataSet, Data, DataVarBound


class Transformer(BaseEstimator, TransformerMixin, ABC):
    """
    Abstract base class to transform an xarray DataArray/Dataset.

    """

    def __init__(
        self,
        sample_name: str = "sample",
        feature_name: str = "feature",
    ):
        self.sample_name = sample_name
        self.feature_name = feature_name

    @abstractmethod
    def get_serialization_attrs(self) -> Dict:
        """Return a dictionary containing the attributes that need to be serialized
        as part of a saved transformer.

        There are limitations on the types of attributes that can be serialized.
        Most simple types (e.g. int, float, str, bool, None) can be, as well as
        DataArrays and dicts of DataArrays. Other nested types (e.g. lists of
        DataArrays) will likely fail.

        """
        return dict()

    @abstractmethod
    def fit(
        self,
        X: Data,
        sample_dims: Optional[Dims] = None,
        feature_dims: Optional[Dims] = None,
        **kwargs,
    ) -> Self:
        """Fit transformer to data.

        Parameters:
        -------------
        X: xr.DataArray | xr.Dataset
            Input data.
        sample_dims: Sequence[Hashable], optional
            Sample dimensions.
        feature_dims: Sequence[Hashable], optional
            Feature dimensions.
        """
        pass

    @abstractmethod
    def transform(self, X: Data) -> Data:
        return X

    def fit_transform(
        self,
        X: Data,
        sample_dims: Optional[Dims] = None,
        feature_dims: Optional[Dims] = None,
        **kwargs,
    ) -> Data:
        return self.fit(X, sample_dims, feature_dims, **kwargs).transform(X)

    @abstractmethod
    def inverse_transform_data(self, X: Data) -> Data:
        return X

    @abstractmethod
    def inverse_transform_components(self, X: Data) -> Data:
        return X

    @abstractmethod
    def inverse_transform_scores(self, X: DataArray) -> DataArray:
        return X

    @abstractmethod
    def inverse_transform_scores_unseen(self, X: DataArray) -> DataArray:
        return X

    def _serialize_data(self, key: str, data: DataArray) -> DataSet:
        # Make sure the DataArray has some name so we can create a string mapping
        if data.name is None:
            data.name = key

        multiindexes = {}
        if data.name in data.coords:
            # Create coords-based datasets and note multiindexes
            if isinstance(data.to_index(), pd.MultiIndex):
                multiindexes[data.name] = [n for n in data.to_index().names]
            ds = xr.Dataset(coords={data.name: data})
        else:
            # Create data-based datasets
            ds = xr.Dataset(data_vars={data.name: data})

        # Drop multiindexes and record for later
        ds = ds.reset_index(list(multiindexes.keys()))
        ds.attrs["multiindexes"] = multiindexes
        ds.attrs["name_map"] = {key: data.name}

        return ds

    def serialize(self) -> DataTree:
        """Serialize a transformer to a DataTree."""
        dt = DataTree()
        params = self.get_params()
        attrs = self.get_serialization_attrs()

        # Set initialization params as tree level attrs
        dt.attrs["params"] = params

        # Serialize each transformer attribute
        for key, attr in attrs.items():
            if isinstance(attr, xr.DataArray):
                # attach data to data_vars or coords
                ds = self._serialize_data(key, attr)
                dt[key] = DataTree(name=key, data=ds)
                dt.attrs[key] = "_is_node"
            elif isinstance(attr, dict) and any(
                [isinstance(val, xr.DataArray) for val in attr.values()]
            ):
                # attach dict of data as branching tree
                dt_attr = DataTree()
                for k, v in attr.items():
                    ds = self._serialize_data(k, v)
                    dt_attr[k] = DataTree(name=k, data=ds)
                dt[key] = dt_attr
                dt.attrs[key] = "_is_tree"
            else:
                # attach simple types as dataset attrs
                dt.attrs[key] = attr

        return dt

    def _deserialize_data_node(self, key: str, ds: xr.Dataset) -> DataArray:
        # Rebuild multiindexes
        ds = ds.set_index(ds.attrs.get("multiindexes", {}))
        # Extract the DataArray or coord from the Dataset
        data_key = ds.attrs["name_map"][key]
        data = ds[data_key]
        return data

    @classmethod
    def deserialize(cls, dt: DataTree) -> Self:
        """Deserialize a saved transformer from a DataTree."""
        # Create the object from params
        params = dt.attrs.pop("params")
        transformer = cls(**params)

        # Set attributes
        for key, attr in dt.attrs.items():
            if attr == "_is_node":
                data = transformer._deserialize_data_node(key, dt[key])
                setattr(transformer, key, data)
            elif attr == "_is_tree":
                data = {}
                for k, v in dt[key].items():
                    data[k] = transformer._deserialize_data_node(k, dt[key][k])
                setattr(transformer, key, data)
            else:
                setattr(transformer, key, attr)

        return transformer
