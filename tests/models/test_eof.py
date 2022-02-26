import numpy as np
import pytest
import warnings
from numpy.testing import assert_allclose

from xeofs.models._eof_base import _EOF_base
from xeofs.models import EOF

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


@pytest.mark.parametrize('method, norm, weights', [
    ('EOF', False, None),
    ('EOF', True, None)
])
def test_solution(method, norm, weights, reference_solution, sample_array):
    # Compare pandas implementation against reference solution
    experiment = reference_solution.get_experiment(
        method=method, norm=norm, weights=weights
    )
    reference = experiment.get_results()
    sample_data = sample_array[:, ~np.isnan(sample_array).all(axis=0)]

    model = EOF(sample_data, norm=norm)
    model.solve()
    assert_allclose(model.singular_values(), reference['singular_values'])
    assert_allclose(model.explained_variance(), reference['explained_variance'])
    assert_allclose(model.explained_variance_ratio(), reference['explained_variance_ratio'])
    assert_allclose(model.eofs(), reference['eofs'])
    assert_allclose(model.pcs(), reference['pcs'])


@pytest.mark.parametrize('n_modes', (1, 5, None))
def test_n_modes(n_modes, sample_array):
    # Number of modes is defined by minimum of sample and feature number
    data_no_nan = sample_array[:, ~np.isnan(sample_array).all(axis=0)]
    base = _EOF_base(data_no_nan, n_modes=n_modes)

    ref_n_modes = min(data_no_nan.shape) if n_modes is None else n_modes

    assert base.n_modes == ref_n_modes


def test_eofs_as_correlation(sample_array):
    # Correlation coefficients are between -1 and 1
    # p values are between 0 and 1
    data_no_nan = sample_array[:, ~np.isnan(sample_array).all(axis=0)]
    model = EOF(data_no_nan)
    model.solve()
    corr, pvals = model.eofs_as_correlation()
    assert (abs(corr) <= 1).all()
    assert (pvals >= 0).all()
    assert (pvals <= 1).all()


@pytest.mark.parametrize('norm', [False, True])
def test_reconstruct_X(norm, sample_array):
    # Data and reconstructed data are close.
    model = EOF(sample_array, norm=norm)
    model.solve()
    Xrec = model.reconstruct_X()
    np.testing.assert_allclose(Xrec, sample_array)


@pytest.mark.parametrize('norm, scaling', [
    (False, 0),
    (False, 1),
    (False, 2),
    (True, 0),
    (True, 1),
    (True, 2),
])
def test_project_onto_eofs(norm, scaling, sample_array):
    # Projection of original data and PCs are the same.
    model = EOF(sample_array, norm=norm)
    model.solve()
    pcs = model.pcs(scaling=scaling)
    projections = model.project_onto_eofs(sample_array, scaling=scaling)
    np.testing.assert_allclose(projections, pcs)
