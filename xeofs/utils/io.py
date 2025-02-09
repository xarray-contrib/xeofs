from ast import literal_eval
from typing import Any

import numpy as np
import xarray as xr


def write_model_tree(
    dt: xr.DataTree, path: str, overwrite: bool = False, engine: str = "zarr", **kwargs
):
    """Write a DataTree to a file."""
    write_mode = "w" if overwrite else "w-"
    if engine in ["netcdf4", "h5netcdf"]:
        dt = _sanitize_attrs_nc(dt)
        dt.to_netcdf(path, engine=engine, **kwargs)
    elif engine == "zarr":
        dt.to_zarr(path, mode=write_mode, **kwargs)
    else:
        raise ValueError(f"Unknown engine {engine}")


def open_model_tree(path: str, engine: str = "zarr", **kwargs) -> xr.DataTree:
    """Open a DataTree from a file."""
    if engine == "zarr" and "chunks" not in kwargs:
        kwargs["chunks"] = {}
    dt = xr.open_datatree(path, engine=engine, **kwargs)
    if engine in ["netcdf4", "h5netcdf"]:
        dt = _desanitize_attrs_nc(dt)
    return dt


def insert_placeholders(dt: xr.DataTree) -> xr.DataTree:
    """Insert placeholders for data that we don't want to compute."""
    for node in dt.subtree:
        if not node.attrs.get("allow_compute", True):
            dt[node.path] = xr.DataTree(
                xr.Dataset(
                    data_vars={
                        node.name: xr.DataArray(np.nan, attrs={"placeholder": True})
                    },
                    attrs={"allow_compute": False, "placeholder": True},
                )
            )
    return dt


def _sanitize_attrs_nc(dt: xr.DataTree) -> xr.DataTree:
    """Sanitize both node-level and variable-level attrs to strings for netcdf."""
    sanitized_types = (dict, list, bool, type(None))
    for node in dt.subtree:
        for key, attr in node.attrs.items():
            if isinstance(attr, sanitized_types):
                node.attrs[key] = str(attr)
        for v in node.variables:
            for key, attr in node[v].attrs.items():
                if isinstance(attr, sanitized_types):
                    node[v].attrs[key] = str(attr)
    return dt


def _should_desanitize(attr: Any) -> bool:
    if isinstance(attr, str):
        if (
            (attr[0] == "{" and attr[-1] == "}")
            or (attr[0] == "[" and attr[-1] == "]")
            or (attr in ["True", "False"])
            or (attr == "None")
        ):
            return True
    return False


def _desanitize_attrs_nc(dt: xr.DataTree) -> xr.DataTree:
    """Desanitize both node-level and variable-level attrs from strings for netcdf."""
    for node in dt.subtree:
        for key, attr in node.attrs.items():
            if _should_desanitize(attr):
                node.attrs[key] = literal_eval(attr)
        for v in node.variables:
            for key, attr in node[v].attrs.items():
                if _should_desanitize(attr):
                    node[v].attrs[key] = literal_eval(attr)
    return dt
