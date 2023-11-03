from typing import Dict

import xarray as xr


def save_to_file(ds: xr.Dataset, path: str, storage_format: str = "netcdf", **kwargs):
    """Save a model serialized to a Dataset, handling different storage backends."""
    if storage_format == "netcdf":
        ds = sanitize_ds(ds)
        ds.to_netcdf(path, **kwargs)
    elif storage_format == "zarr":
        ds.to_zarr(path, **kwargs)


def load_from_file(path: str, storage_format: str = "netcdf", **kwargs) -> xr.Dataset:
    """Load a serialized model object from a file, handling different storage backends."""
    if storage_format == "netcdf":
        ds = xr.open_dataset(path, **kwargs)
        ds = desanitize_ds(ds)
    elif storage_format == "zarr":
        ds = xr.open_zarr(path, **kwargs)
    return ds


def _sanitize_attrs(attrs: Dict) -> Dict:
    """Cast unsafe types to strings for netcdf serialization."""
    sanitize_attrs = {}
    for k, v in attrs.items():
        if isinstance(v, bool):
            sanitize_attrs[k] = str(v)
        elif v is None:
            sanitize_attrs[k] = "None"
        else:
            sanitize_attrs[k] = v
    return sanitize_attrs


def _desanitize_attrs(attrs: Dict) -> Dict:
    """Re-cast sanitized attributes to original types."""
    desanitize_attrs = {}
    for k, v in attrs.items():
        if v == "True":
            desanitize_attrs[k] = True
        elif v == "False":
            desanitize_attrs[k] = False
        elif v == "None":
            desanitize_attrs[k] = None
        else:
            desanitize_attrs[k] = v
    return desanitize_attrs


def sanitize_ds(ds: xr.Dataset) -> xr.Dataset:
    """Cast unsafe types to strings for netcdf serialization."""
    ds.attrs = _sanitize_attrs(ds.attrs)
    for v in ds.data_vars:
        ds[v].attrs = _sanitize_attrs(ds[v].attrs)
    return ds


def desanitize_ds(ds: xr.Dataset) -> xr.Dataset:
    """Re-cast sanitized attributes to original types."""
    ds.attrs = _desanitize_attrs(ds.attrs)
    for v in ds.data_vars:
        ds[v].attrs = _desanitize_attrs(ds[v].attrs)
    return ds
