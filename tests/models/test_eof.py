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

    # Assert preprocessor has been initialized
    assert hasattr(eof, 'preprocessor')


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
])
def test_eof_fit(dim, mock_data_array):
    '''Tests the fit method of the EOF class'''

    eof = EOF()
    eof.fit(mock_data_array, dim)

    # Assert the required attributes have been set
    assert hasattr(eof, 'preprocessor')
    assert hasattr(eof, 'data')


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
    # Explained variance must be positive
    assert (explained_variance > 0).all()


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
    # Explained variance ratio must be positive
    assert (explained_variance_ratio > 0).all(), 'The explained variance ratio must be positive'
    # The sum of the explained variance ratio must be <= 1
    assert explained_variance_ratio.sum() <= 1 + 1e-5, 'The sum of the explained variance ratio must be <= 1'


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
def test_eof_transform(dim, mock_data_array):
    '''Test projecting new unseen data onto the components (EOFs/eigenvectors)'''

    # Create a xarray DataArray with random data
    model = EOF(n_modes=5)
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
    np.testing.assert_allclose(scores.sel(mode=slice(1, 3)), projections.sel(mode=slice(1, 3)), rtol=1e-2)


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