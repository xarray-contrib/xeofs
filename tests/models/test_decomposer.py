import numpy as np
import xarray as xr
import pytest
from dask.array import Array as DaskArray  # type: ignore
from sklearn.utils.extmath import randomized_svd as svd
from scipy.sparse.linalg import svds as complex_svd  # type: ignore
from dask.array.linalg import svd_compressed as dask_svd
from xeofs.models.decomposer import Decomposer


@pytest.fixture
def decomposer():
    return Decomposer(n_modes=2, random_state=42)


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


def test_init(decomposer):
    assert decomposer.n_modes == 2
    assert decomposer.solver_kwargs["random_state"] == 42


def test_fit(decomposer, mock_data_array):
    decomposer.fit(mock_data_array)
    assert "scores_" in decomposer.__dict__
    assert "singular_values_" in decomposer.__dict__
    assert "components_" in decomposer.__dict__


def test_fit_dask(mock_dask_data_array):
    # The Dask SVD solver has no parameter 'random_state' but 'seed' instead,
    # so let's create a new decomposer for this case
    decomposer = Decomposer(n_modes=2, seed=42)
    decomposer.fit(mock_dask_data_array)
    assert "scores_" in decomposer.__dict__
    assert "singular_values_" in decomposer.__dict__
    assert "components_" in decomposer.__dict__


def test_fit_complex(decomposer, mock_complex_data_array):
    decomposer.fit(mock_complex_data_array)
    assert "scores_" in decomposer.__dict__
    assert "singular_values_" in decomposer.__dict__
    assert "components_" in decomposer.__dict__
