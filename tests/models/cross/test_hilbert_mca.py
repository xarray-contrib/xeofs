import pytest
import xarray as xr

from xeofs.cross import HilbertMCA


@pytest.fixture
def mca_model():
    return HilbertMCA(n_modes=3)


def test_initialization():
    mca = HilbertMCA(n_modes=1)
    assert mca is not None


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_fit(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    assert hasattr(mca_model, "preprocessor1")
    assert hasattr(mca_model, "preprocessor2")
    assert hasattr(mca_model, "data")


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_squared_covariance_fraction(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    squared_covariance_fraction = mca_model.squared_covariance_fraction()
    assert isinstance(squared_covariance_fraction, xr.DataArray)
    assert (squared_covariance_fraction > 0).all()
    assert all(
        squared_covariance_fraction <= 1
    ), "Squared covariance fraction is greater than 1"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_covariance_fraction(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    cf = mca_model.covariance_fraction_CD95()
    assert isinstance(cf, xr.DataArray)
    assert all(cf <= 1), "Squared covariance fraction is greater than 1"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    components = mca_model.components()
    assert isinstance(components, tuple), "components is not a tuple"
    assert len(components) == 2, "components list does not have 2 elements"
    assert isinstance(components[0], xr.DataArray), "components[0] is not a DataArray"
    assert isinstance(components[1], xr.DataArray), "components[1] is not a DataArray"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_scores(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    scores = mca_model.scores()
    assert isinstance(scores, tuple), "scores is not a tuple"
    assert len(scores) == 2, "scores list does not have 2 elements"
    assert isinstance(scores[0], xr.DataArray), "scores[0] is not a DataArray"
    assert isinstance(scores[1], xr.DataArray), "scores[1] is not a DataArray"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components_amplitude(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    components = mca_model.components_amplitude()
    assert isinstance(components, tuple), "components is not a tuple"
    assert len(components) == 2, "components list does not have 2 elements"
    assert isinstance(components[0], xr.DataArray), "components[0] is not a DataArray"
    assert isinstance(components[1], xr.DataArray), "components[1] is not a DataArray"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components_phase(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    components = mca_model.components_phase()
    assert isinstance(components, tuple), "components is not a tuple"
    assert len(components) == 2, "components list does not have 2 elements"
    assert isinstance(components[0], xr.DataArray), "components[0] is not a DataArray"
    assert isinstance(components[1], xr.DataArray), "components[1] is not a DataArray"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_scores_amplitude(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    scores = mca_model.scores_amplitude()
    assert isinstance(scores, tuple), "scores is not a tuple"
    assert len(scores) == 2, "scores list does not have 2 elements"
    assert isinstance(scores[0], xr.DataArray), "scores[0] is not a DataArray"
    assert isinstance(scores[1], xr.DataArray), "scores[1] is not a DataArray"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_scores_phase(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    scores = mca_model.scores_phase()
    assert isinstance(scores, tuple), "scores is not a tuple"
    assert len(scores) == 2, "scores list does not have 2 elements"
    assert isinstance(scores[0], xr.DataArray), "scores[0] is not a DataArray"
    assert isinstance(scores[1], xr.DataArray), "scores[1] is not a DataArray"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_fit_empty_data(dim):
    mca = HilbertMCA()
    with pytest.raises(ValueError):
        mca.fit(xr.DataArray(), xr.DataArray(), dim)


@pytest.mark.parametrize(
    "dim",
    [
        ("invalid_dim"),
    ],
)
def test_fit_invalid_dims(mca_model, mock_data_array, dim):
    with pytest.raises(ValueError):
        mca_model.fit(mock_data_array, mock_data_array, dim)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_transform_not_implemented(mca_model, mock_data_array, dim):
    with pytest.raises(NotImplementedError):
        mca_model.transform(mock_data_array, mock_data_array)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_fit_with_dataset(mca_model, mock_dataset, dim):
    mca_model.fit(mock_dataset, mock_dataset, dim)
    assert hasattr(mca_model, "preprocessor1")
    assert hasattr(mca_model, "preprocessor2")
    assert hasattr(mca_model, "data")


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_fit_with_datalist(mca_model, mock_data_array_list, dim):
    mca_model.fit(mock_data_array_list, mock_data_array_list, dim)
    assert hasattr(mca_model, "preprocessor1")
    assert hasattr(mca_model, "preprocessor2")
    assert hasattr(mca_model, "data")
