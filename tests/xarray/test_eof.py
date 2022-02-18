import numpy as np
import pytest
import warnings
from numpy.testing import assert_allclose

from xeofs.xarray import EOF


warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


@pytest.mark.parametrize('method, norm, weights', [
    ('EOF', False, None),
    ('EOF', True, None)
])
def test_solution(method, norm, weights, reference_solution, sample_DataArray):
    # Compare numpy implementation against reference solution
    experiment = reference_solution.get_experiment(
        method=method, norm=norm, weights=weights
    )
    reference = experiment.get_results()
    sample_data = sample_DataArray.dropna('loc')

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
    # xarray.DataArray are accepted only.
    with pytest.raises(Exception):
        _ = EOF(sample_array)


def test_coslat_weighting(sample_DataArray):
    # Coslat weighting does not raise an error
    da = sample_DataArray.unstack().rename({'y': 'lat'})
    model = EOF(da, weights='coslat')
    model.solve()
    _ = model.eofs()


def test_invalid_coslat_weighting(sample_DataArray):
    # Coslat weighting does not raise an error
    invalid_da1 = sample_DataArray
    invalid_da2 = sample_DataArray.unstack()
    invalid_da3 = sample_DataArray.rename({'loc': 'lat'})

    with pytest.raises(Exception):
        _ = EOF(invalid_da1, weights='coslat')
    with pytest.raises(Exception):
        _ = EOF(invalid_da2, weights='coslat')
    with pytest.raises(Exception):
        _ = EOF(invalid_da3, weights='coslat')
