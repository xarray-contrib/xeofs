from typing import Dict
from dask.diagnostics.progress import ProgressBar

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

    def compute(self, verbose=False):
        for k, v in self.items():
            if self._allow_compute[k]:
                if verbose:
                    with ProgressBar():
                        self[k] = v.compute()
                else:
                    self[k] = v.compute()

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
