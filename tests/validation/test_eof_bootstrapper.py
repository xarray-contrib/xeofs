import numpy as np
import pytest
import xarray as xr

from xeofs.single import EOF
from xeofs.validation import EOFBootstrapper


# If the data is defined as a fixture (e.g., mock_data_array),
# we can't directly parametrize it with the @pytest.mark.parametrize decorator.
# A way to go around this is to create a fixture that uses another fixture.
# This might be a bit complex, but it should work
@pytest.fixture(params=["mock_data_array", "mock_dataset", "mock_data_array_list"])
def data(request):
    # Here, request.param would be a fixture name as a string
    return request.getfixturevalue(request.param)


@pytest.fixture
def dim(request):
    return request.param


@pytest.fixture
def eof_model(data, dim):
    eof = EOF()
    eof.fit(data, dim)
    return eof


def test_init():
    bootstrapper = EOFBootstrapper(n_bootstraps=3, seed=7)
    assert bootstrapper._params["n_bootstraps"] == 3
    assert bootstrapper._params["seed"] == 7


@pytest.mark.parametrize(
    "data, dim",
    [
        ("mock_data_array", "time"),
        ("mock_data_array", ("lon", "lat")),
        ("mock_data_array", ("lat", "lon")),
        ("mock_dataset", "time"),
        ("mock_dataset", ("lon", "lat")),
        ("mock_dataset", ("lat", "lon")),
        ("mock_data_array_list", "time"),
        ("mock_data_array_list", ("lon", "lat")),
        ("mock_data_array_list", ("lat", "lon")),
    ],
    indirect=["data", "dim"],
)
def test_fit(eof_model):
    """Bootstrapping creates DataArrays with expected dims and coords"""
    bootstrapper = EOFBootstrapper(n_bootstraps=3, seed=7)
    bootstrapper.fit(eof_model)

    # DataArrays are created
    assert isinstance(
        bootstrapper.data["explained_variance"], xr.DataArray
    ), "explained variance is not a DataArray"
    assert isinstance(
        bootstrapper.data["components"], xr.DataArray
    ), "components is not a DataArray"
    assert isinstance(
        bootstrapper.data["scores"], xr.DataArray
    ), "scores is not a DataArray"

    # DataArrays have expected dims
    expected_dims = set(eof_model.data["explained_variance"].dims)
    expected_dims.add("n")
    true_dims = set(bootstrapper.data["explained_variance"].dims)
    err_message = (
        f"explained variance dimensions are {true_dims} instead of {expected_dims}"
    )
    assert true_dims == expected_dims, err_message

    expected_dims = set(eof_model.data["components"].dims)
    expected_dims.add("n")
    true_dims = set(bootstrapper.data["components"].dims)
    err_message = f"components dimensions are {true_dims} instead of {expected_dims}"
    assert true_dims == expected_dims, err_message

    expected_dims = set(eof_model.data["scores"].dims)
    expected_dims.add("n")
    true_dims = set(bootstrapper.data["scores"].dims)
    err_message = f"scores dimensions are {true_dims} instead of {expected_dims}"
    assert true_dims == expected_dims, err_message

    # DataArrays have expected coords
    ref_da = eof_model.data["explained_variance"]
    test_da = bootstrapper.data["explained_variance"]
    for dim, coords in ref_da.coords.items():
        assert test_da[dim].equals(
            coords
        ), f"explained variance coords for {dim} are not equal"

    ref_da = eof_model.data["components"]
    test_da = bootstrapper.data["components"]
    for dim, coords in ref_da.coords.items():
        assert test_da[dim].equals(coords), f"components coords for {dim} are not equal"

    ref_da = eof_model.data["scores"]
    test_da = bootstrapper.data["scores"]
    for dim, coords in ref_da.coords.items():
        assert test_da[dim].equals(coords), f"scores coords for {dim} are not equal"


@pytest.mark.parametrize(
    "data, dim",
    [
        ("mock_data_array", "time"),
        ("mock_data_array", ("lon", "lat")),
        ("mock_data_array", ("lat", "lon")),
        ("mock_dataset", "time"),
        ("mock_dataset", ("lon", "lat")),
        ("mock_dataset", ("lat", "lon")),
        ("mock_data_array_list", "time"),
        ("mock_data_array_list", ("lon", "lat")),
        ("mock_data_array_list", ("lat", "lon")),
    ],
    indirect=["data", "dim"],
)
def test_seed(eof_model):
    """Bootstrapping creates DataArrays with expected dims and coords"""
    bootstrapper1 = EOFBootstrapper(n_bootstraps=5, seed=7)
    bootstrapper1.fit(eof_model)
    expvar1 = bootstrapper1.explained_variance()

    bootstrapper2 = EOFBootstrapper(n_bootstraps=5, seed=7)
    bootstrapper2.fit(eof_model)
    expvar2 = bootstrapper2.explained_variance()

    assert np.allclose(
        expvar1.values, expvar2.values
    ), "bootstrapping with the same seed does not produce the same results"


@pytest.mark.parametrize(
    "data, dim",
    [
        ("mock_data_array", "time"),
        ("mock_data_array", ("lon", "lat")),
        ("mock_data_array", ("lat", "lon")),
        ("mock_dataset", "time"),
        ("mock_dataset", ("lon", "lat")),
        ("mock_dataset", ("lat", "lon")),
        ("mock_data_array_list", "time"),
        ("mock_data_array_list", ("lon", "lat")),
        ("mock_data_array_list", ("lat", "lon")),
    ],
    indirect=["data", "dim"],
)
def test_explained_variance(eof_model):
    """Bootstrapping creates DataArrays expected dims and coords"""
    bootstrapper = EOFBootstrapper(n_bootstraps=3)
    bootstrapper.fit(eof_model)

    expvar = bootstrapper.explained_variance()
    assert isinstance(expvar, xr.DataArray), "explained variance is not a DataArray"

    # DataArrays have expected dims
    ref = eof_model.explained_variance()
    expected_dims = set(ref.dims)
    expected_dims.add("n")
    true_dims = set(expvar.dims)
    err_message = (
        f"explained variance dimensions are {true_dims} instead of {expected_dims}"
    )
    assert true_dims == expected_dims, err_message

    # DataArrays have expected coords
    for dim, coords in ref.coords.items():
        assert expvar[dim].equals(
            coords
        ), f"explained variance coords for {dim} are not equal"


@pytest.mark.parametrize(
    "data, dim",
    [
        ("mock_data_array", "time"),
        ("mock_data_array", ("lon", "lat")),
        ("mock_data_array", ("lat", "lon")),
        ("mock_dataset", "time"),
        ("mock_dataset", ("lon", "lat")),
        ("mock_dataset", ("lat", "lon")),
        ("mock_data_array_list", "time"),
        ("mock_data_array_list", ("lon", "lat")),
        ("mock_data_array_list", ("lat", "lon")),
    ],
    indirect=["data", "dim"],
)
def test_explained_variance_ratio(eof_model):
    """Bootstrapping creates DataArrays expected dims and coords"""
    bootstrapper = EOFBootstrapper(n_bootstraps=3)
    bootstrapper.fit(eof_model)

    expvar = bootstrapper.explained_variance_ratio()
    assert isinstance(
        expvar, xr.DataArray
    ), "explained variance ratio is not a DataArray"

    # DataArrays have expected dims
    ref = eof_model.explained_variance_ratio()
    expected_dims = set(ref.dims)
    expected_dims.add("n")
    true_dims = set(expvar.dims)
    err_message = f"explained variance ratio dimensions are {true_dims} instead of {expected_dims}"
    assert true_dims == expected_dims, err_message

    # DataArrays have expected coords
    for dim, coords in ref.coords.items():
        assert expvar[dim].equals(
            coords
        ), f"explained variance ratio coords for {dim} are not equal"


@pytest.mark.parametrize(
    "data, dim, expected_type",
    [
        ("mock_data_array", "time", xr.DataArray),
        ("mock_data_array", ("lon", "lat"), xr.DataArray),
        ("mock_data_array", ("lat", "lon"), xr.DataArray),
        ("mock_dataset", "time", xr.Dataset),
        ("mock_dataset", ("lon", "lat"), xr.Dataset),
        ("mock_dataset", ("lat", "lon"), xr.Dataset),
    ],
    indirect=["data", "dim"],
)
def test_components(eof_model, expected_type):
    """Bootstrapping components creates expected dims and coords"""
    bootstrapper = EOFBootstrapper(n_bootstraps=3)
    bootstrapper.fit(eof_model)

    components = bootstrapper.components()
    assert isinstance(components, expected_type), f"components is not a {expected_type}"

    # check for expected dimensions
    ref = eof_model.components()
    expected_dims = set(ref.dims)
    expected_dims.add("n")
    true_dims = set(components.dims)
    err_message = f"components dimensions are {true_dims} instead of {expected_dims}"
    assert true_dims == expected_dims, err_message

    # check for expected coords
    for dim, coords in ref.coords.items():
        assert components[dim].equals(
            coords
        ), f"components coords for {dim} are not equal"


@pytest.mark.parametrize(
    "data, dim, expected_type",
    [
        ("mock_data_array_list", "time", list),
        ("mock_data_array_list", ("lon", "lat"), list),
        ("mock_data_array_list", ("lat", "lon"), list),
    ],
    indirect=["data", "dim"],
)
def test_components_list(eof_model, expected_type):
    """Bootstrapping components creates expected dims and coords"""
    bootstrapper = EOFBootstrapper(n_bootstraps=3)
    bootstrapper.fit(eof_model)

    comps_list = bootstrapper.components()
    assert isinstance(comps_list, expected_type), f"components is not a {expected_type}"

    # check for expected dimensions
    ref_list = eof_model.components()
    for ref, comps in zip(ref_list, comps_list):
        expected_dims = set(ref.dims)
        expected_dims.add("n")
        true_dims = set(comps.dims)
        err_message = (
            f"components dimensions are {true_dims} instead of {expected_dims}"
        )
        assert true_dims == expected_dims, err_message

    # check for expected coords
    for ref, comps in zip(ref_list, comps_list):
        for dim, coords in ref.coords.items():
            assert comps[dim].equals(
                coords
            ), f"components coords for {dim} are not equal"


@pytest.mark.parametrize(
    "data, dim",
    [
        ("mock_data_array", "time"),
        ("mock_data_array", ("lon", "lat")),
        ("mock_data_array", ("lat", "lon")),
        ("mock_dataset", "time"),
        ("mock_dataset", ("lon", "lat")),
        ("mock_dataset", ("lat", "lon")),
        ("mock_data_array_list", "time"),
        ("mock_data_array_list", ("lon", "lat")),
        ("mock_data_array_list", ("lat", "lon")),
    ],
    indirect=["data", "dim"],
)
def test_scores(eof_model):
    """Bootstrapping scores creates expected dims and coords"""
    bootstrapper = EOFBootstrapper(n_bootstraps=3)
    bootstrapper.fit(eof_model)

    scores = bootstrapper.scores()
    assert isinstance(scores, xr.DataArray), "scores is not a xr.DataArray"

    # check for expected dimensions
    ref = eof_model.scores()
    expected_dims = set(ref.dims)
    expected_dims.add("n")
    true_dims = set(scores.dims)
    err_message = f"scores dimensions are {true_dims} instead of {expected_dims}"
    assert true_dims == expected_dims, err_message

    # check for expected coords
    for dim, coords in ref.coords.items():
        assert scores[dim].equals(coords), f"scores coords for {dim} are not equal"
