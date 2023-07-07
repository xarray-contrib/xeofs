import numpy as np
import xarray as xr
import pytest
import dask.array as da
from numpy.testing import assert_allclose

from xeofs.models.eof import EOF


def test_eof_initialization():
    '''Tests the initialization of the EOF class'''
    eof = EOF(n_modes=5, standardize=True, use_coslat=True)

    # Assert parameters are correctly stored in the _params attribute
    assert eof._params == {'n_modes': 5, 'standardize': True, 'use_coslat': True, 'use_weights': False}

    # Assert correct values are stored in the _scaling_params attribute
    assert eof._scaling_params == {'with_std': True, 'with_coslat': True, 'with_weights': False}


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
])
def test_eof_fit(dim, mock_data_array):
    '''Tests the fit method of the EOF class'''

    eof = EOF()
    eof.fit(mock_data_array, dim)

    # Assert that data has been preprocessed
    assert isinstance(eof.data, xr.DataArray)

    # Assert the required attributes have been set
    assert eof._total_variance is not None
    assert eof._singular_values is not None
    assert eof._explained_variance is not None
    assert eof._explained_variance_ratio is not None
    assert eof._components is not None
    assert eof._scores is not None


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
])
def test_eof_singular_values(dim, mock_data_array):
    '''Tests the singular_values method of the EOF class'''

    eof = EOF()
    eof.fit(mock_data_array, dim)

    # Test singular_values method
    singular_values = eof.singular_values()
    assert isinstance(singular_values, xr.DataArray)


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
])
def test_eof_explained_variance(dim, mock_data_array):
    '''Tests the explained_variance method of the EOF class'''
    eof = EOF()
    eof.fit(mock_data_array, dim)

    # Test explained_variance method
    explained_variance = eof.explained_variance()
    assert isinstance(explained_variance, xr.DataArray)


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
])
def test_eof_explained_variance_ratio(dim, mock_data_array):
    '''Tests the explained_variance_ratio method of the EOF class'''
    eof = EOF()
    eof.fit(mock_data_array, dim)

    # Test explained_variance_ratio method
    explained_variance_ratio = eof.explained_variance_ratio()
    assert isinstance(explained_variance_ratio, xr.DataArray)


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
])
def test_eof_components(dim, mock_data_array):
    '''Tests the components method of the EOF class'''
    eof = EOF()
    eof.fit(mock_data_array, dim)

    # Test components method
    components = eof.components()
    assert isinstance(components, xr.DataArray)


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
])
def test_eof_scores(dim, mock_data_array):
    '''Tests the scores method of the EOF class'''
    eof = EOF()
    eof.fit(mock_data_array, dim)

    # Test scores method
    scores = eof.scores()
    assert isinstance(scores, xr.DataArray)


def test_eof_get_params():
    '''Tests the get_params method of the EOF class'''
    eof = EOF(n_modes=5, standardize=True, use_coslat=True)

    # Test get_params method
    params = eof.get_params()
    assert isinstance(params, dict)
    assert params == {'n_modes': 5, 'standardize': True, 'use_coslat': True, 'use_weights': False}


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
])
def test_eof_compute(dim, mock_data_array):
    '''Tests the compute method of the EOF class'''
    
    dask_mock_data_array = mock_data_array.chunk({'time': 1})
    
    eof = EOF()
    eof.fit(dask_mock_data_array, dim)

   # Assert that the attributes are indeed Dask arrays before computation
    assert isinstance(eof._total_variance.data, da.Array)  #type: ignore
    assert isinstance(eof._singular_values.data, da.Array)  #type: ignore
    assert isinstance(eof._explained_variance.data, da.Array)  #type: ignore
    assert isinstance(eof._explained_variance_ratio.data, da.Array)  #type: ignore
    assert isinstance(eof._components.data, da.Array)  #type: ignore
    assert isinstance(eof._scores.data, da.Array)  #type: ignore

    # Test compute method
    eof.compute()

    # Assert the attributes are no longer Dask arrays after computation
    assert not isinstance(eof._total_variance.data, da.Array)  #type: ignore
    assert not isinstance(eof._singular_values.data, da.Array)  #type: ignore
    assert not isinstance(eof._explained_variance.data, da.Array)  #type: ignore
    assert not isinstance(eof._explained_variance_ratio.data, da.Array)  #type: ignore
    assert not isinstance(eof._components.data, da.Array)  #type: ignore
    assert not isinstance(eof._scores.data, da.Array)  #type: ignore


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
])
def test_eof_transform(dim, mock_data_array):
    '''Test projecting new unseen data onto the components (EOFs/eigenvectors)'''

    # Create a xarray DataArray with random data
    model = EOF(n_modes=2)
    model.fit(mock_data_array, dim)
    scores = model.scores()

    # Create a new xarray DataArray with random data
    new_data = mock_data_array

    projections = model.transform(new_data)

    # Check that the projection has the right dimensions
    assert projections.dims == scores.dims, 'Projection has wrong dimensions' #type: ignore

    # Check that the projection has the right data type
    assert isinstance(projections, xr.DataArray), 'Projection is not a DataArray'

    # Check that the projection has the right name
    assert projections.name == 'scores', 'Projection has wrong name'

    # Check that the projection's data is the same as the scores
    np.testing.assert_allclose(scores.data, projections.data)


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
])
def test_eof_inverse_transform(dim, mock_data_array):
    '''Test inverse_transform method in EOF class.'''

    # instantiate the EOF class with necessary parameters
    eof = EOF(n_modes=3, standardize=True)
    
    # fit the EOF model
    eof.fit(mock_data_array, dim=dim)

    # Test with scalar
    mode = 1
    reconstructed_data = eof.inverse_transform(mode)
    assert isinstance(reconstructed_data, xr.DataArray)
    
    # Test with slice
    mode = slice(1, 2)
    reconstructed_data = eof.inverse_transform(mode)
    assert isinstance(reconstructed_data, xr.DataArray)

    # Test with array of tick labels
    mode = np.array([1, 3])
    reconstructed_data = eof.inverse_transform(mode)
    assert isinstance(reconstructed_data, xr.DataArray)

    # Check that the reconstructed data has the same dimensions as the original data
    assert set(reconstructed_data.dims) == set(mock_data_array.dims)