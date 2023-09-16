from typing import Dict, Optional

import numpy as np
import pytest
import warnings
import xarray as xr
import pandas as pd

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


# =============================================================================
# Synthetic data
# =============================================================================
def generate_synthetic_dataarray(
    n_sample=1, n_feature=1, has_multiindex=False, has_nan=None, is_dask=False, seed=0
):
    """Create synthetic DataArray.

    Parameters:
    ------------
    n_sample: int
        Number of sample dimensions.
    n_dims_feature: int
        Number of feature dimensions.
    has_multiindex: bool, default=False
        If True, the data will have a multiindex.
    has_nan: [None, "isolated", "fulldim"], default=None
        If specified, the data will contain NaNs.
    is_dask: bool, default=False
        If True, the data will be a dask array.
    seed: int, default=0
        Seed for the random number generator.

    Returns:
    ---------
    data: xr.DataArray
        Synthetic data.

    """
    rng = np.random.default_rng(seed)

    # Create dimensions
    sample_dims = [f"sample{i}" for i in range(n_sample)]
    feature_dims = [f"feature{i}" for i in range(n_feature)]
    all_dims = feature_dims + sample_dims

    # Create coordinates
    coords = {}
    for i, dim in enumerate(all_dims):
        if has_multiindex:
            coords[dim] = pd.MultiIndex.from_arrays(
                [np.arange(6 - i), np.arange(6 - i)],
                names=[f"index{i}a", f"index{i}b"],
            )
        else:
            coords[dim] = np.arange(6 + i)

    # Get data shape
    shape = tuple([len(coords[dim]) for dim in all_dims])

    # Create data
    noise = rng.normal(5, 3, size=shape)
    signal = 2 * np.sin(np.linspace(0, 2 * np.pi, shape[-1]))
    signal = np.broadcast_to(signal, shape)
    data = signal + noise
    data = xr.DataArray(data, dims=all_dims, coords=coords)

    # Add NaNs
    if has_nan is not None:
        if has_nan == "isolated":
            isolated_point = {dim: 0 for dim in all_dims}
            data.loc[isolated_point] = np.nan
        elif has_nan == "fulldim":
            fulldim_point = {dim: 0 for dim in feature_dims}
            data.loc[fulldim_point] = np.nan
        else:
            raise ValueError(f"Invalid value for has_nan: {has_nan}")

    if is_dask:
        data = data.chunk({"sample_0": 1})

    return data


def generate_synthetic_dataset(
    n_variables=1,
    n_sample=1,
    n_feature=1,
    has_multiindex=False,
    has_nan=None,
    is_dask=False,
    seed=0,
):
    """Create synthetic Dataset.

    Parameters:
    ------------
    n_variables: int
        Number of variables.
    n_sample: int
        Number of sample dimensions.
    n_dims_feature: int
        Number of feature dimensions.
    has_multiindex: bool, default=False
        If True, the data will have a multiindex.
    has_nan: [None, "isolated", "fulldim"], default=None
        If specified, the data will contain NaNs.
    is_dask: bool, default=False
        If True, the data will be a dask array.
    seed: int, default=0
        Seed for the random number generator.

    Returns:
    ---------
    data: xr.Dataset
        Synthetic data.

    """
    data = generate_synthetic_dataarray(
        n_sample, n_feature, has_multiindex, has_nan, is_dask, seed
    )
    dataset = xr.Dataset({"var0": data})
    seed += 1

    for n in range(1, n_variables):
        data_n = generate_synthetic_dataarray(
            n_sample=n_sample,
            n_feature=n_feature,
            has_multiindex=has_multiindex,
            has_nan=has_nan,
            is_dask=is_dask,
            seed=seed,
        )
        dataset[f"var{n}"] = data_n
        seed += 1
    return dataset


def generate_list_of_synthetic_dataarrays(
    n_arrays=1,
    n_sample=1,
    n_feature=1,
    has_multiindex=False,
    has_nan=None,
    is_dask=False,
    seed=0,
):
    """Create synthetic Dataset.

    Parameters:
    ------------
    n_arrays: int
        Number of DataArrays.
    n_sample: int
        Number of sample dimensions.
    n_dims_feature: int
        Number of feature dimensions.
    has_multiindex: bool, default=False
        If True, the data will have a multiindex.
    has_nan: [None, "isolated", "fulldim"], default=None
        If specified, the data will contain NaNs.
    is_dask: bool, default=False
        If True, the data will be a dask array.
    seed: int, default=0
        Seed for the random number generator.

    Returns:
    ---------
    data: xr.Dataset
        Synthetic data.

    """
    data_arrays = []
    for n in range(n_arrays):
        data_n = generate_synthetic_dataarray(
            n_sample=n_sample,
            n_feature=n_feature,
            has_multiindex=has_multiindex,
            has_nan=has_nan,
            is_dask=is_dask,
            seed=seed,
        )
        data_arrays.append(data_n)
        seed += 1
    return data_arrays


@pytest.fixture
def synthetic_dataarray(request):
    data = generate_synthetic_dataarray(*request.param)
    return data


@pytest.fixture
def synthetic_dataset(request):
    data = generate_synthetic_dataset(*request.param)
    return data


@pytest.fixture
def synthetic_dataarray_list(request):
    data = generate_list_of_synthetic_dataarrays(*request.param)
    return data


# =============================================================================
# Input data
# =============================================================================
@pytest.fixture
def mock_data_array():
    rng = np.random.default_rng(7)
    noise = rng.normal(5, 3, size=(25, 5, 4))
    signal = 2 * np.sin(np.linspace(0, 2 * np.pi, 25))[:, None, None]
    return xr.DataArray(
        signal + noise,
        dims=("time", "lat", "lon"),
        coords={
            "time": xr.date_range("2001", "2025", freq="YS"),
            "lat": [20.0, 30.0, 40.0, 50.0, 60.0],
            "lon": [-10.0, 0.0, 10.0, 20.0],
        },
        name="t2m",
        attrs=dict(description="mock_data"),
    )


@pytest.fixture
def mock_dataset(mock_data_array):
    t2m = mock_data_array
    prcp = t2m**2
    return xr.Dataset({"t2m": t2m, "prcp": prcp})


@pytest.fixture
def mock_data_array_list(mock_data_array):
    da1 = mock_data_array
    da2 = mock_data_array**2
    da3 = mock_data_array**3
    return [da1, da2, da3]


@pytest.fixture
def mock_data_array_isolated_nans(mock_data_array):
    invalid_data = mock_data_array.copy()
    invalid_data.loc[dict(time="2001", lat=30.0, lon=-10.0)] = np.nan
    return invalid_data


@pytest.fixture
def mock_data_array_full_dimensional_nans(mock_data_array):
    valid_data = mock_data_array.copy()
    valid_data.loc[dict(lat=30.0)] = np.nan
    valid_data.loc[dict(time="2002")] = np.nan
    return valid_data


@pytest.fixture
def mock_data_array_boundary_nans(mock_data_array):
    valid_data = mock_data_array.copy(deep=True)
    valid_data.loc[dict(lat=30.0)] = np.nan
    valid_data.loc[dict(time="2001")] = np.nan
    return valid_data


@pytest.fixture
def mock_dask_data_array(mock_data_array):
    return mock_data_array.chunk({"lon": 2, "lat": 2, "time": -1})


# =============================================================================
# Intermediate data
# =============================================================================
@pytest.fixture
def sample_input_data():
    """Create a sample input data."""
    return xr.DataArray(np.random.rand(10, 20), dims=("sample", "feature"))


@pytest.fixture
def sample_components():
    """Create a sample components."""
    return xr.DataArray(np.random.rand(20, 5), dims=("feature", "mode"))


@pytest.fixture
def sample_scores():
    """Create a sample scores."""
    return xr.DataArray(np.random.rand(10, 5), dims=("sample", "mode"))


@pytest.fixture
def sample_exp_var():
    return xr.DataArray(
        np.random.rand(10),
        dims=("mode",),
        coords={"mode": np.arange(10)},
        name="explained_variance",
    )


@pytest.fixture
def sample_total_variance(sample_exp_var):
    return sample_exp_var.sum()


@pytest.fixture
def sample_idx_modes_sorted(sample_exp_var):
    return sample_exp_var.argsort()[::-1]


@pytest.fixture
def sample_norm():
    return xr.DataArray(np.random.rand(10), dims=("mode",))


@pytest.fixture
def sample_squared_covariance():
    return xr.DataArray(np.random.rand(10), dims=("mode",))


@pytest.fixture
def sample_total_squared_covariance(sample_squared_covariance):
    return sample_squared_covariance.sum("mode")


@pytest.fixture
def sample_rotation_matrix():
    """Create a sample rotation matrix."""
    return xr.DataArray(np.random.rand(5, 5), dims=("mode_m", "mode_n"))


@pytest.fixture
def sample_phi_matrix():
    """Create a sample phi matrix."""
    return xr.DataArray(np.random.rand(5, 5), dims=("mode_m", "mode_n"))


@pytest.fixture
def sample_modes_sign():
    """Create a sample modes sign."""
    return xr.DataArray(np.random.choice([-1, 1], size=5), dims=("mode",))
