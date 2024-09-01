import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Literal

import dask.base
import xarray as xr
from typing_extensions import Self

from ._version import __version__
from .utils.data_types import DataArray
from .utils.io import insert_placeholders, open_model_tree, write_model_tree
from .utils.xarray_utils import data_is_dask

try:
    from xarray.core.datatree import DataTree  # type: ignore
except ImportError:
    from datatree import DataTree

# Ignore warnings from numpy casting with additional coordinates
warnings.filterwarnings("ignore", message=r"^invalid value encountered in cast*")

xr.set_options(keep_attrs=True)


class BaseModel(ABC):
    """
    Abstract base class for an xeofs model.

    Provides basic functionality for lazy model evaluation, serialization, deserialization and saving/loading models.

    """

    def __init__(self):
        # Define model parameters
        self._params = {}

        # Define analysis-relevant meta data
        self.attrs = {"model": "BaseModel"}
        self.attrs.update(
            {
                "software": "xeofs",
                "version": __version__,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        self.attrs.update(self._params)

    @abstractmethod
    def get_serialization_attrs(self) -> dict:
        """Get the attributes to serialize."""
        raise NotImplementedError

    def compute(self, **kwargs):
        """Compute and load delayed model results.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to `dask.compute()`.
        """
        # find and compute all dask arrays simultaneously to allow dask to optimize the
        # shared graph and avoid duplicate i/o and computations
        dt = self.serialize()

        data_objs = {
            k: v
            for k, v in dt.to_dict().items()
            if data_is_dask(v) and v.attrs.get("allow_compute", True)
        }

        (data_objs,) = dask.base.compute(data_objs, **kwargs)

        for k, v in data_objs.items():
            dt[k] = DataTree(v)

        # then rebuild the trained model from the computed results
        self._deserialize_attrs(dt)

        self._post_compute()

    def _post_compute(self):
        pass

    def get_params(self) -> dict[str, Any]:
        """Get the model parameters."""
        return self._params

    def serialize(self) -> DataTree:
        """Serialize a complete model with its preprocessor."""
        # Create a root node for this object with its params as attrs
        ds_root = xr.Dataset(attrs=dict(params=self.get_params()))
        dt = DataTree(data=ds_root, name=type(self).__name__)

        # Retrieve the tree representation of each attached object, or set basic attrs
        for key, attr in self.get_serialization_attrs().items():
            if hasattr(attr, "serialize"):
                dt[key] = attr.serialize()
                dt.attrs[key] = "_is_tree"
            else:
                dt.attrs[key] = attr

        return dt

    def save(
        self,
        path: str,
        overwrite: bool = False,
        save_data: bool = False,
        engine: Literal["zarr", "netcdf4", "h5netcdf"] = "zarr",
        **kwargs,
    ):
        """Save the model.

        Parameters
        ----------
        path : str
            Path to save the model.
        overwrite: bool, default=False
            Whether or not to overwrite the existing path if it already exists.
            Ignored unless `engine="zarr"`.
        save_data : str
            Whether or not to save the full input data along with the fitted components.
        engine : {"zarr", "netcdf4", "h5netcdf"}, default="zarr"
            Xarray backend engine to use for writing the saved model.
        **kwargs
            Additional keyword arguments to pass to `DataTree.to_netcdf()` or `DataTree.to_zarr()`.

        """
        self.compute()

        dt = self.serialize()

        # Remove any raw data arrays at this stage
        if not save_data:
            dt = insert_placeholders(dt)

        write_model_tree(dt, path, overwrite=overwrite, engine=engine, **kwargs)

    @classmethod
    def deserialize(cls, dt: DataTree) -> Self:
        """Deserialize the model and its preprocessors from a DataTree."""
        # Recreate the model with parameters set by root level attrs
        model = cls(**dt.attrs["params"])
        model._deserialize_attrs(dt)
        return model

    def _deserialize_attrs(self, dt: DataTree):
        """Set the necessary attributes of the model from a DataTree."""
        for key, attr in dt.attrs.items():
            if key == "params":
                continue
            elif attr == "_is_tree":
                deserialized_obj = getattr(self, str(key)).deserialize(dt[str(key)])
            else:
                deserialized_obj = attr
            setattr(self, str(key), deserialized_obj)

    @classmethod
    def load(
        cls,
        path: str,
        engine: Literal["zarr", "netcdf4", "h5netcdf"] = "zarr",
        **kwargs,
    ) -> Self:
        """Load a saved model.

        Parameters
        ----------
        path : str
            Path to the saved model.
        engine : {"zarr", "netcdf4", "h5netcdf"}, default="zarr"
            Xarray backend engine to use for reading the saved model.
        **kwargs
            Additional keyword arguments to pass to `open_datatree()`.

        Returns
        -------
        model : BaseModel
            The loaded model.

        """
        dt = open_model_tree(path, engine=engine, **kwargs)
        model = cls.deserialize(dt)
        return model

    def _validate_loaded_data(self, X: DataArray):
        """Optionally check the loaded data for placeholders."""
        pass
