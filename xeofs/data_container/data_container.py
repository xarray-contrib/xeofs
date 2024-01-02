from typing import Dict
from typing_extensions import Self

import dask
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from datatree import DataTree

from ..utils.data_types import DataArray


class DataContainer(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._allow_compute = dict({k: True for k in self.keys()})

    def add(self, data: DataArray, name: str, allow_compute: bool = True) -> None:
        data.name = name
        super().__setitem__(name, data)
        self._allow_compute[name] = True if allow_compute else False

    def __setitem__(self, __key: str, __value: DataArray) -> None:
        super().__setitem__(__key, __value)
        self._allow_compute[__key] = self._allow_compute.get(__key, True)

    def __getitem__(self, __key: str) -> DataArray:
        try:
            return super().__getitem__(__key)
        except KeyError:
            raise KeyError(
                f"Cannot find data '{__key}'. Please fit the model first by calling .fit()."
            )

    def serialize(self) -> DataTree:
        dt = DataTree(name="data")
        for key, data in self.items():
            if not data.name:
                data.name = key
            dt[key] = DataTree(data)
            dt[key].attrs = {key: "_is_node", "allow_compute": self._allow_compute[key]}

        return dt

    @classmethod
    def deserialize(cls, dt: DataTree) -> Self:
        container = cls()
        for key, node in dt.items():
            container[key] = node[key]
            container._allow_compute[key] = node.attrs["allow_compute"]
        return container

    def compute(self, verbose=False, **kwargs):
        computed_data = {k: v for k, v in self.items() if self._allow_compute[k]}
        if verbose:
            with ProgressBar():
                (computed_data,) = dask.compute(computed_data, **kwargs)
        else:
            (computed_data,) = dask.compute(computed_data, **kwargs)
        for k, v in computed_data.items():
            self[k] = v

    def _validate_attrs(self, attrs: Dict) -> Dict:
        """Convert any boolean and None values to strings"""
        for key, value in attrs.items():
            if isinstance(value, bool):
                attrs[key] = str(value)
            elif value is None:
                attrs[key] = "None"

        return attrs

    def set_attrs(self, attrs: Dict):
        attrs = self._validate_attrs(attrs)
        for key in self.keys():
            self[key].attrs = attrs
