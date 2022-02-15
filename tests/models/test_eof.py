import numpy as np
import pytest
import warnings
from numpy.testing import assert_allclose

from xeofs.models import EOF


warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


@pytest.mark.parametrize('method, norm, weights', [
    ('EOF', False, None),
    ('EOF', True, None)
])
def test_solution(method, norm, weights, reference_solution, sample_array):
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
