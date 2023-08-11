import numpy as np
import xarray as xr
import pytest
from dask.array import Array as DaskArray  # type: ignore
from sklearn.utils.extmath import randomized_svd as svd
from scipy.sparse.linalg import svds as complex_svd  # type: ignore
from dask.array.linalg import svd_compressed as dask_svd
from xeofs.models.decomposer import CrossDecomposer


@pytest.fixture
def mock_data_array(mock_data_array):
    return mock_data_array.stack(sample=("time",), feature=("lat", "lon")).dropna(
        "feature"
    )


@pytest.fixture
def mock_dask_data_array(mock_data_array):
    return mock_data_array.chunk({"sample": 2})


@pytest.fixture
def mock_complex_data_array(mock_data_array):
    return mock_data_array * (1 + 1j)


@pytest.fixture
def test_complex_dask_data_array(mock_complex_data_array):
    return mock_complex_data_array.chunk({"sample": 2})


@pytest.fixture
def cross_decomposer():
    return CrossDecomposer(n_modes=2, random_state=42)


def test_init(cross_decomposer):
    assert cross_decomposer.n_modes == 2
    assert cross_decomposer.solver_kwargs["random_state"] == 42


def test_fit(cross_decomposer, mock_data_array):
    cross_decomposer.fit(mock_data_array, mock_data_array)
    assert "singular_vectors1_" in cross_decomposer.__dict__
    assert "singular_values_" in cross_decomposer.__dict__
    assert "singular_vectors2_" in cross_decomposer.__dict__


def test_fit_complex(cross_decomposer, mock_complex_data_array):
    cross_decomposer.fit(mock_complex_data_array, mock_complex_data_array)
    assert "singular_vectors1_" in cross_decomposer.__dict__
    assert "singular_values_" in cross_decomposer.__dict__
    assert "singular_vectors2_" in cross_decomposer.__dict__


def test_fit_same_samples(cross_decomposer, mock_data_array):
    with pytest.raises(ValueError):
        cross_decomposer.fit(mock_data_array, mock_data_array.isel(sample=slice(1, 3)))


def test_fit_dask(mock_dask_data_array):
    # The Dask SVD solver has no parameter 'random_state' but 'seed' instead,
    # so let's create a new decomposer for this case
    cross_decomposer = CrossDecomposer(n_modes=2, seed=42)
    cross_decomposer.fit(mock_dask_data_array, mock_dask_data_array)
    assert "singular_vectors1_" in cross_decomposer.__dict__
    assert "singular_values_" in cross_decomposer.__dict__
    assert "singular_vectors2_" in cross_decomposer.__dict__