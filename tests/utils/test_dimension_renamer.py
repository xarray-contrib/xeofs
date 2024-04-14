import xarray as xr
import pytest

from xeofs.utils.dimension_renamer import DimensionRenamer

import pandas as pd


@pytest.fixture
def da_simple():
    # Basic DataArray with single dimension
    return xr.DataArray([1, 2, 3], coords={"dim0": ["a", "b", "c"]}, dims=["dim0"])


@pytest.fixture
def da_multi():
    # DataArray with MultiIndex
    arrays = [
        pd.Index(["a", "b", "c"], name="dim1a"),
        pd.Index([1, 2, 3], name="dim1b"),
    ]
    multi_index = pd.MultiIndex.from_arrays(arrays)
    return xr.DataArray([1, 2, 3], coords={"dim1": multi_index}, dims=["dim1"])


def test_simple_dim_rename(da_simple):
    renamer = DimensionRenamer("dim0", "_suffix")
    da_new = renamer.fit_transform(da_simple)
    assert "dim0_suffix" in da_new.dims

    # Inverse transform
    da_orig = renamer.inverse_transform(da_new)
    assert "dim0" in da_orig.dims


def test_multiindex_dim_rename(da_multi):
    renamer = DimensionRenamer("dim1", "_suffix")
    da_new = renamer.fit_transform(da_multi)
    assert "dim1_suffix" in da_new.dims
    assert "dim1a_suffix" in da_new.coords["dim1_suffix"].coords.keys()
    assert "dim1b_suffix" in da_new.coords["dim1_suffix"].coords.keys()

    # Inverse transform
    da_orig = renamer.inverse_transform(da_new)
    assert "dim1" in da_orig.dims
    assert "dim1a" in da_orig.coords["dim1"].coords.keys()
    assert "dim1b" in da_orig.coords["dim1"].coords.keys()


def test_fit_without_transform(da_simple):
    renamer = DimensionRenamer("dim0", "_suffix")
    renamer.fit(da_simple)
    assert hasattr(renamer, "dims_mapping")
    assert renamer.dims_mapping == {"dim0": "dim0_suffix"}


def test_incorrect_dim_name(da_simple):
    with pytest.raises(KeyError):
        renamer = DimensionRenamer("nonexistent_dim", "_suffix")
        renamer.fit_transform(da_simple)


def test_empty_suffix(da_simple):
    renamer = DimensionRenamer("dim0", "")
    da_new = renamer.fit_transform(da_simple)
    assert "dim0" in da_new.dims
