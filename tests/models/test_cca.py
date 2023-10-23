import numpy as np
import xarray as xr
import pytest
import dask.array as da
from cca_zoo.linear import MCCA as ReferenceCCA
from cca_zoo.linear import PCACCA as ReferenceCCA2
from numpy.testing import assert_allclose
from ..conftest import generate_list_of_synthetic_dataarrays

from xeofs.models.cca import CCA


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


@pytest.mark.parametrize(
    "c",
    [
        (0.0),
        (0.5),
        (1.0),
    ],
)
def test_solution(c):
    """Check numerical results with cca-zoo reference implementation"""

    dalist = generate_list_of_synthetic_dataarrays(
        2, 1, 1, "index", "no_nan", "no_dask"
    )
    # Ensure that the numpy 2D arrays  is in the correct format
    Xlist = [X.transpose("sample0", "feature0").data for X in dalist]

    cca = CCA(n_modes=2, pca=False, c=c)
    cca.fit(dalist, dim="sample0")
    comps = cca.components()
    scores = cca.scores()

    # Compare with cca-zoo
    # cca-zoo requires centered data
    Xlist = [X - X.mean(0) for X in Xlist]
    cca_ref = ReferenceCCA(latent_dimensions=2, c=c)
    scores_ref = cca_ref.fit_transform(Xlist)
    comps_ref = cca_ref.factor_loadings(Xlist)

    for i in range(len(scores)):
        assert_allclose(abs(comps[i]), abs(comps_ref[i].T), rtol=1e-5)
        assert_allclose(abs(scores[i]), abs(scores_ref[i].T), rtol=1e-5)
