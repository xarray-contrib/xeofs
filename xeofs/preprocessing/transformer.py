from abc import ABC, abstractmethod

import pandas as pd
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin
from typing_extensions import Self

try:
    from xarray.core.datatree import DataTree
except ImportError:
    from datatree import DataTree

from ..utils.data_types import Data, DataArray, DataSet, Dims


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
    def get_serialization_attrs(self) -> dict:
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
        sample_dims: Dims | None = None,
        feature_dims: Dims | None = None,
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
        sample_dims: Dims | None = None,
        feature_dims: Dims | None = None,
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

    def _serialize_data(self, key: str, data: Data) -> DataSet:
        multiindexes = {}
        name_map = None
        if isinstance(data, xr.Dataset):
            # Keep Dataset as is
            ds = data
        else:
            # Convert DataArray to Dataset
            coords = {}
            data_vars = {}
            if data.name in data.coords:
                # Convert a coord-like DataArray to Dataset and note multiindexes
                if isinstance(data.to_index(), pd.MultiIndex):
                    multiindexes[data.name] = [n for n in data.to_index().names]
                coords[data.name] = data
            else:
                # Make sure the DataArray has some name so we can create a string mapping
                if data.name is None:
                    data.name = key
                data_vars[data.name] = data
            ds = xr.Dataset(data_vars=data_vars, coords=coords)
            name_map = data.name

        # Drop multiindexes and record for later
        ds = ds.reset_index(list(multiindexes.keys()))
        ds.attrs["multiindexes"] = multiindexes
        ds.attrs["name_map"] = {key: name_map}

        return ds

    def serialize(self) -> DataTree:
        """Serialize a transformer to a DataTree."""
        return self._serialize()

    def _serialize(self) -> DataTree:
        """Serialize a transformer to a DataTree. Use an internal
        method so we can override the public one in subclasesses but
        still use this."""
        dt = DataTree()
        params = self.get_params()
        attrs = self.get_serialization_attrs()

        # Set initialization params as tree level attrs
        dt.attrs["params"] = params

        # Serialize each transformer attribute
        for key, attr in attrs.items():
            if isinstance(attr, (xr.DataArray, xr.Dataset)):
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

    def _deserialize_data_node(self, key: str, dt: DataTree) -> DataArray:
        # Rebuild multiindexes
        dt = dt.set_index(dt.attrs.get("multiindexes", {}))
        # Extract the DataArray or coord from the Dataset
        data_key = dt.attrs["name_map"][key]
        if data_key is not None:
            return dt[data_key]
        else:
            return dt.ds

    @classmethod
    def deserialize(cls, dt: DataTree) -> Self:
        """Deserialize a saved transformer from a DataTree."""
        return cls._deserialize(dt)

    @classmethod
    def _deserialize(cls, dt: DataTree) -> Self:
        """Deserialize a saved transformer from a DataTree. Use an internal
        method so we can override the public one in subclasesses but
        still use this."""
        # Create the object from params
        transformer = cls(**dt.attrs["params"])

        # Set attributes
        for key, attr in dt.attrs.items():
            if key == "params":
                continue
            elif attr == "_is_node":
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
