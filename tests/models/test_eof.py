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


def test_abstract_class(sample_array):
    # Abstract class methods are not implemented
    data_no_nan = sample_array[:, ~np.isnan(sample_array).all(axis=0)]
    base = _EOF_base(data_no_nan)
    with pytest.raises(Exception):
        base.solve()
    with pytest.raises(Exception):
        base.singular_values()
    with pytest.raises(Exception):
        base.explained_variance()
    with pytest.raises(Exception):
        base.explained_variance_ratio()
    with pytest.raises(Exception):
        base.eofs()
    with pytest.raises(Exception):
        base.pcs()
