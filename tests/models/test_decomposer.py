import numpy as np
import xarray as xr
import pytest
from dask.array import Array as DaskArray  # type: ignore
from sklearn.utils.extmath import randomized_svd as svd
from scipy.sparse.linalg import svds as complex_svd  # type: ignore
from dask.array.linalg import svd_compressed as dask_svd
from xeofs.models.decomposer import Decomposer, CrossDecomposer


@pytest.fixture
def decomposer():
    return Decomposer(n_modes=2, n_iter=3, random_state=42, verbose=False)


@pytest.fixture
def cross_decomposer():
    return CrossDecomposer(n_modes=2, n_iter=3, random_state=42, verbose=False)


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


def test_decomposer_init(decomposer):
    assert decomposer.params["n_modes"] == 2
    assert decomposer.params["n_iter"] == 3
    assert decomposer.params["random_state"] == 42
    assert decomposer.params["verbose"] == False


def test_cross_decomposer_init(cross_decomposer):
    assert cross_decomposer.params["n_modes"] == 2
    assert cross_decomposer.params["n_iter"] == 3
    assert cross_decomposer.params["random_state"] == 42
    assert cross_decomposer.params["verbose"] == False


def test_decomposer_fit(decomposer, mock_data_array):
    decomposer.fit(mock_data_array)
    assert "scores_" in decomposer.__dict__
    assert "singular_values_" in decomposer.__dict__
    assert "components_" in decomposer.__dict__


def test_decomposer_fit_dask(decomposer, mock_dask_data_array):
    decomposer.fit(mock_dask_data_array)
    assert "scores_" in decomposer.__dict__
    assert "singular_values_" in decomposer.__dict__
    assert "components_" in decomposer.__dict__


def test_decomposer_fit_complex(decomposer, mock_complex_data_array):
    decomposer.fit(mock_complex_data_array)
    assert "scores_" in decomposer.__dict__
    assert "singular_values_" in decomposer.__dict__
    assert "components_" in decomposer.__dict__


def test_cross_decomposer_fit(cross_decomposer, mock_data_array):
    cross_decomposer.fit(mock_data_array, mock_data_array)
    assert "singular_vectors1_" in cross_decomposer.__dict__
    assert "singular_values_" in cross_decomposer.__dict__
    assert "singular_vectors2_" in cross_decomposer.__dict__


def test_cross_decomposer_fit_complex(cross_decomposer, mock_complex_data_array):
    cross_decomposer.fit(mock_complex_data_array, mock_complex_data_array)
    assert "singular_vectors1_" in cross_decomposer.__dict__
    assert "singular_values_" in cross_decomposer.__dict__
    assert "singular_vectors2_" in cross_decomposer.__dict__


def test_cross_decomposer_fit_dask(cross_decomposer, mock_dask_data_array):
    cross_decomposer.fit(mock_dask_data_array, mock_dask_data_array)
    assert "singular_vectors1_" in cross_decomposer.__dict__
    assert "singular_values_" in cross_decomposer.__dict__
    assert "singular_vectors2_" in cross_decomposer.__dict__


def test_cross_decomposer_fit_same_samples(cross_decomposer, mock_data_array):
    with pytest.raises(ValueError):
        cross_decomposer.fit(mock_data_array, mock_data_array.isel(sample=slice(1, 3)))
