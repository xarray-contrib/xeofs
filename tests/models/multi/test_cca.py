import pytest

from xeofs.multi import CCA


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_fit(dim, mock_data_array_list):
    """Tests the fit method of the CCA class"""

    cca = CCA()
    cca.fit(mock_data_array_list, dim)

    # Assert the required attributes have been set
    assert hasattr(cca, "preprocessors")
    assert hasattr(cca, "data")


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components(dim, mock_data_array_list):
    """Tests the components method of the CCA class"""

    cca = CCA()
    cca.fit(mock_data_array_list, dim)

    comps = cca.components()
    assert isinstance(comps, list)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_scores(dim, mock_data_array_list):
    """Tests the components method of the CCA class"""

    cca = CCA()
    cca.fit(mock_data_array_list, dim)

    scores = cca.scores()
    assert isinstance(scores, list)
