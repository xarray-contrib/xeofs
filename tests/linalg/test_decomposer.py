import dask
import numpy as np
import pytest
import xarray as xr
from dask.array import Array as DaskArray  # type: ignore
from dask.callbacks import Callback

from xeofs.linalg.decomposer import Decomposer

from ..utilities import data_is_dask


def compute_max_exp_var(singular_values, data):
    """Compute the maximal cumulative explained variance by all components."""

    total_variance = data.var("sample", ddof=1).sum("feature")
    explained_variance = singular_values**2 / (data.sample.size - 1)
    explained_variance_ratio = explained_variance / total_variance
    explained_variance_ratio_cumsum = explained_variance_ratio.cumsum("mode")
    return explained_variance_ratio_cumsum.isel(mode=-1).item()


@pytest.fixture
def decomposer():
    return Decomposer(n_modes=2, random_state=42)


@pytest.fixture
def mock_data_array(mock_data_array):
    data2d = mock_data_array.stack(sample=("time",), feature=("lat", "lon")).dropna(
        "feature"
    )
    return data2d - data2d.mean("sample")


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


@pytest.mark.parametrize("shape", [(12, 8), (8, 12), (8, 8)])
@pytest.mark.parametrize(
    "chunks",
    [(3, 3), (3, -1), (-1, 3), (-1, -1)],
    ids=["both-axes", "rows", "columns", "single-block"],
)
@pytest.mark.parametrize("solver", ["auto", "full"])
@pytest.mark.parametrize("compute", [False, True])
def test_fit_dask_exact_matches_numpy(shape, chunks, solver, compute, monkeypatch):
    """Exact SVD must preserve results and laziness across chunk layouts."""
    dims = ("left_feature", "right_feature")
    data = xr.DataArray(np.random.default_rng(42).normal(size=shape), dims=dims)
    chunked = data.chunk(dict(zip(dims, chunks)))
    original_chunks = chunked.chunks
    expected_numblocks = list(chunked.data.numblocks)
    if all(n > 1 for n in expected_numblocks):
        expected_numblocks[np.argmin(shape)] = 1

    decomposer = Decomposer(n_modes=min(shape), solver=solver, compute=compute)
    original_svd = decomposer._svd
    calls = []

    def record_svd(X, dims, func, kwargs):
        calls.append((X.data.numblocks, func))
        return original_svd(X, dims, func, kwargs)

    monkeypatch.setattr(decomposer, "_svd", record_svd)
    tasks = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):
        decomposer.fit(chunked, dims=dims)

    assert not tasks, "Fitting the exact Dask SVD must not execute the graph"
    assert chunked.chunks == original_chunks
    assert calls == [(tuple(expected_numblocks), np.linalg.svd)]
    assert all(
        isinstance(result.data, DaskArray)
        for result in (decomposer.U_, decomposer.s_, decomposer.V_)
    )

    reference = Decomposer(n_modes=min(shape), solver=solver, compute=compute)
    reference.fit(data, dims=dims)
    U, s, V = dask.compute(decomposer.U_, decomposer.s_, decomposer.V_)
    for actual, expected in zip((U, s, V), (reference.U_, reference.s_, reference.V_)):
        xr.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        (U.values * s.values) @ V.values.T, data.values, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("shape", [(499, 4), (4, 499)])
def test_fit_dask_full_rechunks_just_below_size_limit(shape):
    """The automatic-rechunking cutoff is strictly below 500 on both axes."""
    data = xr.DataArray(
        np.random.default_rng(42).normal(size=shape), dims=("sample", "feature")
    ).chunk({"sample": shape[0] // 2, "feature": shape[1] // 2})
    decomposer = Decomposer(n_modes=2, solver="full", compute=False)

    tasks = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):
        decomposer.fit(data)

    assert not tasks
    assert isinstance(decomposer.s_.data, DaskArray)


@pytest.mark.parametrize("shape", [(500, 4), (4, 500)])
@pytest.mark.parametrize("compute", [False, True])
def test_fit_dask_full_rejects_large_fully_chunked_input(shape, compute, monkeypatch):
    """Reject redistribution before rechunking or executing any Dask tasks."""
    data = xr.DataArray(
        np.random.default_rng(42).normal(size=shape), dims=("sample", "feature")
    ).chunk({"sample": shape[0] // 2, "feature": shape[1] // 2})
    original_chunks = data.chunks
    original_rechunk = DaskArray.rechunk
    rechunks = []

    def record_rechunk(array, *args, **kwargs):
        rechunks.append(array.shape)
        return original_rechunk(array, *args, **kwargs)

    monkeypatch.setattr(DaskArray, "rechunk", record_rechunk)
    decomposer = Decomposer(n_modes=2, solver="full", compute=compute)
    tasks = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):
        with pytest.raises(
            ValueError, match="large Dask matrices.*requires explicit rechunking"
        ):
            decomposer.fit(data)

    assert not rechunks
    assert not tasks
    assert data.chunks == original_chunks


@pytest.mark.parametrize("shape", [(500, 4), (4, 500)])
@pytest.mark.parametrize("layout", ["numpy", "sample", "feature", "single-block"])
def test_fit_large_full_preserves_supported_layouts(shape, layout):
    """The cutoff must not restrict NumPy or already-supported Dask layouts."""
    data = xr.DataArray(
        np.random.default_rng(42).normal(size=shape), dims=("sample", "feature")
    )
    if layout == "numpy":
        X = data
    elif layout == "single-block":
        X = data.chunk({"sample": -1, "feature": -1})
    else:
        X = data.chunk({layout: data.sizes[layout] // 2})
    original_chunks = X.chunks

    decomposer = Decomposer(n_modes=2, solver="full", compute=False)
    decomposer.fit(X)
    reference = Decomposer(n_modes=2, solver="full")
    reference.fit(data)

    assert X.chunks == original_chunks
    for name in ("U_", "s_", "V_"):
        xr.testing.assert_allclose(
            getattr(decomposer, name).compute(),
            getattr(reference, name),
            rtol=1e-12,
            atol=1e-12,
        )


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


@pytest.mark.parametrize(
    "target_variance, solver",
    [
        (0.1, "randomized"),
        (0.5, "randomized"),
        (0.9, "randomized"),
        (0.99, "randomized"),
        (0.1, "full"),
        (0.5, "full"),
        (0.9, "full"),
        (0.99, "full"),
    ],
)
def test_decompose_via_variance_threshold(mock_data_array, target_variance, solver):
    """Test that the decomposer returns the correct number of modes to explain the target variance."""
    decomposer = Decomposer(
        n_modes=target_variance, solver=solver, init_rank_reduction=0.9
    )
    decomposer.fit(mock_data_array)
    s = decomposer.s_

    # Compute total variance and test whether variance threshold is reached
    max_explained_variance_ratio = compute_max_exp_var(s, mock_data_array)
    assert (
        max_explained_variance_ratio >= target_variance
    ), f"Expected >= {target_variance:.2f}, got {max_explained_variance_ratio:2f}"

    # We still get a truncated version of the SVD
    assert s.mode.size < min(mock_data_array.shape)


def test_raise_warning_for_low_init_rank_reduction(mock_data_array):
    target_variance = 0.5
    init_rank_reduction = 0.1
    decomposer = Decomposer(
        n_modes=target_variance, init_rank_reduction=init_rank_reduction
    )
    warn_msg = "Dataset has .* components, explaining .* of the variance. However, .*explained variance was requested. Please consider increasing `init_rank_reduction`"
    with pytest.warns(UserWarning, match=warn_msg):
        decomposer.fit(mock_data_array)


def test_compute_at_least_one_component(mock_data_array):
    """"""
    target_variance = 0.5
    init_rank_reduction = 0.01
    decomposer = Decomposer(
        n_modes=target_variance, init_rank_reduction=init_rank_reduction
    )

    # Warning is raised to indicate that the value of init_rank_reduction is too low
    warn_msg = "`init_rank_reduction=.*` is too low resulting in zero components. One component will be computed instead."
    with pytest.warns(UserWarning, match=warn_msg):
        decomposer.fit(mock_data_array)

    # At least one mode is computed
    s = decomposer.s_
    assert s.mode.size >= 1


@pytest.mark.parametrize(
    "solver",
    ["full", "randomized"],
)
def test_dask_array_based_on_target_variance(mock_dask_data_array, solver):
    target_variance = 0.5
    decomposer = Decomposer(
        n_modes=target_variance, init_rank_reduction=0.9, solver=solver, compute=False
    )

    err_msg = "Estimating the number of modes to keep based on variance is not supported with dask arrays.*"
    with pytest.raises(ValueError, match=err_msg):
        assert data_is_dask(mock_dask_data_array)
        decomposer.fit(mock_dask_data_array)
