import numpy as np
import pytest
import warnings
from numpy.testing import assert_allclose

from xeofs.pandas import EOF


warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


@pytest.mark.parametrize('method, norm, weights', [
    ('EOF', False, None),
    ('EOF', True, None)
])
def test_solution(method, norm, weights, reference_solution, sample_DataFrame):
    # Compare pandas implementation against reference solution
    experiment = reference_solution.get_experiment(
        method=method, norm=norm, weights=weights
    )
    reference = experiment.get_results()
    sample_data = sample_DataFrame.dropna(axis=1)

    model = EOF(sample_data, norm=norm)
    model.solve()
    singular_values = model.singular_values().values.squeeze()
    explained_variance = model.explained_variance().values.squeeze()
    explained_variance_ratio = model.explained_variance_ratio().values.squeeze()
    eofs = model.eofs()
    pcs = model.pcs()
    assert_allclose(singular_values, reference['singular_values'])
    assert_allclose(explained_variance, reference['explained_variance'])
    assert_allclose(explained_variance_ratio, reference['explained_variance_ratio'])
    assert_allclose(eofs, reference['eofs'])
    assert_allclose(pcs, reference['pcs'])


def test_invalid_input_type(sample_array):
    # pandas.DataFrame are accepted only.
    with pytest.raises(Exception):
        _ = EOF(sample_array)
