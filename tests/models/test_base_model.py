import numpy as np
import xarray as xr
import pytest
import dask.array as da
import warnings
from numpy.testing import assert_allclose

from xeofs.models._base_model import EOF

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


@pytest.mark.parametrize('method, standardize, use_weights', [
    ('EOF', False, None),
    ('EOF', True, None)
])
def test_solution(method, standardize, use_weights, reference_solution, test_data):
    # Compare numpy implementation against reference solution
    experiment = reference_solution.get_experiment(
        method=method, norm=standardize, weights=use_weights
    )
    reference = experiment.get_results()

    model = EOF(standardize=standardize)
    model.fit(test_data.transpose('time','x','y'), 'time')
    assert_allclose(model.singular_values(), reference['singular_values'])
    assert_allclose(model.explained_variance(), reference['explained_variance'])
    assert_allclose(model.explained_variance_ratio(), reference['explained_variance_ratio'])
    assert_allclose(model.components().stack(loc=('x', 'y')).dropna('loc').values, reference['eofs'].T)
    assert_allclose(model.scores().values, reference['pcs'].T)



def test_EOF_initialization():
    """Tests the initialization of the EOF class"""
    eof = EOF(n_modes=5, standardize=True, use_coslat=True)

    # Assert parameters are correctly stored in the _params attribute
    assert eof._params == {'n_modes': 5, 'standardize': True, 'use_coslat': True, 'use_weights': False}

    # Assert correct values are stored in the _scaling_params attribute
    assert eof._scaling_params == {'with_std': True, 'with_coslat': True, 'with_weights': False}


def test_EOF_fit(test_data):
    """Tests the fit method of the EOF class"""
    dims = 'time'

    eof = EOF()
    eof.fit(test_data, dims)

    # Assert that data has been preprocessed
    assert isinstance(eof.data, xr.DataArray)

    # Assert the required attributes have been set
    assert eof._total_variance is not None
    assert eof._singular_values is not None
    assert eof._explained_variance is not None
    assert eof._explained_variance_ratio is not None
    assert eof._components is not None
    assert eof._scores is not None


def test_EOF_singular_values(test_data):
    """Tests the singular_values method of the EOF class"""
    dims = 'time'

    eof = EOF()
    eof.fit(test_data, dims)

    # Test singular_values method
    singular_values = eof.singular_values()
    assert isinstance(singular_values, xr.DataArray)


def test_EOF_explained_variance(test_data):
    """Tests the explained_variance method of the EOF class"""
    dims = 'time'

    eof = EOF()
    eof.fit(test_data, dims)

    # Test explained_variance method
    explained_variance = eof.explained_variance()
    assert isinstance(explained_variance, xr.DataArray)

def test_EOF_explained_variance_ratio(test_data):
    """Tests the explained_variance_ratio method of the EOF class"""
    dims = 'time'

    eof = EOF()
    eof.fit(test_data, dims)

    # Test explained_variance_ratio method
    explained_variance_ratio = eof.explained_variance_ratio()
    assert isinstance(explained_variance_ratio, xr.DataArray)


def test_EOF_components(test_data):
    """Tests the components method of the EOF class"""
    dims = 'time'

    eof = EOF()
    eof.fit(test_data, dims)

    # Test components method
    components = eof.components()
    assert isinstance(components, xr.DataArray)


def test_EOF_scores(test_data):
    """Tests the scores method of the EOF class"""
    dims = 'time'

    eof = EOF()
    eof.fit(test_data, dims)

    # Test scores method
    scores = eof.scores()
    assert isinstance(scores, xr.DataArray)


def test_EOF_get_params():
    """Tests the get_params method of the EOF class"""
    eof = EOF(n_modes=5, standardize=True, use_coslat=True)

    # Test get_params method
    params = eof.get_params()
    assert isinstance(params, dict)
    assert params == {'n_modes': 5, 'standardize': True, 'use_coslat': True, 'use_weights': False}


def test_EOF_compute(test_data):
    """Tests the compute method of the EOF class"""
    
    dims = 'time'
    
    dask_test_data = test_data.chunk({'time': 1})
    
    eof = EOF()
    eof.fit(dask_test_data, dims)

   # Assert that the attributes are indeed Dask arrays before computation
    assert isinstance(eof._total_variance.data, da.Array)
    assert isinstance(eof._singular_values.data, da.Array)
    assert isinstance(eof._explained_variance.data, da.Array)
    assert isinstance(eof._explained_variance_ratio.data, da.Array)
    assert isinstance(eof._components.data, da.Array)
    assert isinstance(eof._scores.data, da.Array)

    # Test compute method
    eof.compute()

    # Assert the attributes are no longer Dask arrays after computation
    assert not isinstance(eof._total_variance.data, da.Array)
    assert not isinstance(eof._singular_values.data, da.Array)
    assert not isinstance(eof._explained_variance.data, da.Array)
    assert not isinstance(eof._explained_variance_ratio.data, da.Array)
    assert not isinstance(eof._components.data, da.Array)
    assert not isinstance(eof._scores.data, da.Array)


