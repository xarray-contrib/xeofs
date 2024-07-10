import numpy as np
import pytest
from dask.array import Array as DaskArray  # type: ignore
from xeofs.models.decomposer import Decomposer
from ..utilities import data_is_dask


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
    assert decomposer.random_state == 42


def test_fit_full(mock_data_array):
    decomposer = Decomposer(n_modes=2, solver="full")
    decomposer.fit(mock_data_array)
    assert "U_" in decomposer.__dict__
    assert "s_" in decomposer.__dict__
    assert "V_" in decomposer.__dict__

    # Check that indeed 2 modes are returned
    assert decomposer.U_.shape[1] == 2
    assert decomposer.s_.shape[0] == 2
    assert decomposer.V_.shape[1] == 2


def test_fit_full_matrices(mock_data_array):
    decomposer = Decomposer(
        n_modes=2, solver="full", solver_kwargs={"full_matrices": False}
    )
    decomposer.fit(mock_data_array)
    assert "U_" in decomposer.__dict__
    assert "s_" in decomposer.__dict__
    assert "V_" in decomposer.__dict__

    # Check that indeed 2 modes are returned
    assert decomposer.U_.shape[1] == 2
    assert decomposer.s_.shape[0] == 2
    assert decomposer.V_.shape[1] == 2


def test_fit_randomized(mock_data_array):
    decomposer = Decomposer(n_modes=2, solver="randomized", random_state=42)
    decomposer.fit(mock_data_array)
    assert "U_" in decomposer.__dict__
    assert "s_" in decomposer.__dict__
    assert "V_" in decomposer.__dict__

    # Check that indeed 2 modes are returned
    assert decomposer.U_.shape[1] == 2
    assert decomposer.s_.shape[0] == 2
    assert decomposer.V_.shape[1] == 2


def test_fit_dask_full(mock_dask_data_array):
    # The Dask SVD solver has no parameter 'random_state' but 'seed' instead,
    # so let's create a new decomposer for this case
    decomposer = Decomposer(n_modes=2, solver="full")
    decomposer.fit(mock_dask_data_array)
    assert "U_" in decomposer.__dict__
    assert "s_" in decomposer.__dict__
    assert "V_" in decomposer.__dict__

    # Check if the Dask SVD solver has been used
    assert isinstance(decomposer.U_.data, DaskArray)
    assert isinstance(decomposer.s_.data, DaskArray)
    assert isinstance(decomposer.V_.data, DaskArray)

    # Check that indeed 2 modes are returned
    assert decomposer.U_.shape[1] == 2
    assert decomposer.s_.shape[0] == 2
    assert decomposer.V_.shape[1] == 2


@pytest.mark.parametrize("compute", [True, False])
def test_fit_dask_randomized(mock_dask_data_array, compute):
    # The Dask SVD solver has no parameter 'random_state' but 'seed' instead,
    # so this should be automatically converted depending on the solver
    decomposer = Decomposer(
        n_modes=2, solver="randomized", compute=compute, random_state=42
    )
    decomposer.fit(mock_dask_data_array)
    assert "U_" in decomposer.__dict__
    assert "s_" in decomposer.__dict__
    assert "V_" in decomposer.__dict__

    # Check that indeed 2 modes are returned
    assert decomposer.U_.shape[1] == 2
    assert decomposer.s_.shape[0] == 2
    assert decomposer.V_.shape[1] == 2

    is_dask_before = data_is_dask(mock_dask_data_array)
    U_is_dask_after = data_is_dask(decomposer.U_)
    s_is_dask_after = data_is_dask(decomposer.s_)
    V_is_dask_after = data_is_dask(decomposer.V_)
    # Check if the Dask SVD solver has been used
    assert is_dask_before
    if compute:
        assert not U_is_dask_after
        assert not s_is_dask_after
        assert not V_is_dask_after
    else:
        assert U_is_dask_after
        assert s_is_dask_after
        assert V_is_dask_after


def test_fit_complex(mock_complex_data_array):
    decomposer = Decomposer(n_modes=2, solver="randomized", random_state=42)
    decomposer.fit(mock_complex_data_array)
    assert "U_" in decomposer.__dict__
    assert "s_" in decomposer.__dict__
    assert "V_" in decomposer.__dict__

    # Check that indeed 2 modes are returned
    assert decomposer.U_.shape[1] == 2
    assert decomposer.s_.shape[0] == 2
    assert decomposer.V_.shape[1] == 2

    # Check that U and V are complex
    assert np.iscomplexobj(decomposer.U_.data)
    assert np.iscomplexobj(decomposer.V_.data)


@pytest.mark.parametrize(
    "data",
    ["real", "complex", "dask_real"],
)
def test_random_state(
    data, mock_data_array, mock_complex_data_array, mock_dask_data_array
):
    match data:
        case "real":
            X = mock_data_array
        case "complex":
            X = mock_complex_data_array
        case "dask_real":
            X = mock_dask_data_array
        case _:
            raise ValueError(f"Unrecognized data type '{data}'.")

    decomposer = Decomposer(
        n_modes=2, solver="randomized", random_state=42, compute=True
    )
    decomposer.fit(X)
    U1 = decomposer.U_.data

    # Refit
    decomposer.fit(X)
    U2 = decomposer.U_.data

    # Check that the results are the same
    assert np.all(U1 == U2)
