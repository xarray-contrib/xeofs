import pytest
import numpy as np
import xarray as xr
from dask.array import Array as DaskArray  # type: ignore
from numpy.testing import assert_allclose

from xeofs.models.mca import MCA

@pytest.fixture
def mca_model():
    return MCA()

def test_mca_initialization():
    mca = MCA()
    assert mca is not None


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
])
def test_mca_fit(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    assert mca_model._singular_values is not None
    assert mca_model._explained_variance is not None
    assert mca_model._squared_total_variance is not None
    assert mca_model._singular_vectors1 is not None
    assert mca_model._singular_vectors2 is not None
    assert mca_model._norm1 is not None
    assert mca_model._norm2 is not None


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
])
def test_mca_fit_empty_data(mca_model, dim):
    with pytest.raises(ValueError):
        mca_model.fit(xr.DataArray(), xr.DataArray(), dim)


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
])
def test_mca_fit_invalid_dims(mca_model, mock_data_array, dim):
    with pytest.raises(ValueError):
        mca_model.fit(mock_data_array, mock_data_array, dim=('invalid_dim1', 'invalid_dim2'))


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
])
def test_mca_fit_with_dataset(mca_model, mock_dataset, dim):
    mca_model.fit(mock_dataset, mock_dataset, dim)
    assert mca_model._singular_values is not None


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
])
def test_mca_fit_with_dataarray_list(mca_model, mock_data_array_list, dim):
    mca_model.fit(mock_data_array_list, mock_data_array_list, dim)
    assert mca_model._singular_values is not None


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
])
def test_transform(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    result = mca_model.transform(data1=mock_data_array, data2=mock_data_array)
    assert isinstance(result, list)
    assert isinstance(result[0], xr.DataArray)


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
]) 
def test_inverse_transform(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    # Assuming mode as 1 for simplicity
    Xrec1, Xrec2 = mca_model.inverse_transform(1)
    assert isinstance(Xrec1, xr.DataArray)
    assert isinstance(Xrec2, xr.DataArray)


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
])   
def test_singular_values(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    singular_values = mca_model.singular_values()
    assert isinstance(singular_values, xr.DataArray)


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
])  
def test_explained_variance(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    explained_variance = mca_model.explained_variance()
    assert isinstance(explained_variance, xr.DataArray)


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
])
def test_squared_covariance_fraction(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    squared_covariance_fraction = mca_model.squared_covariance_fraction()
    assert isinstance(squared_covariance_fraction, xr.DataArray)


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
])
def test_components(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    components1, components2 = mca_model.components()
    assert isinstance(components1, xr.DataArray)
    assert isinstance(components2, xr.DataArray)


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
])
def test_scores(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    scores1, scores2 = mca_model.scores()
    assert isinstance(scores1, xr.DataArray)
    assert isinstance(scores2, xr.DataArray)


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
])
def test_homogeneous_patterns(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    patterns, pvals = mca_model.homogeneous_patterns()
    assert isinstance(patterns[0], xr.DataArray)
    assert isinstance(patterns[1], xr.DataArray)
    assert isinstance(pvals[0], xr.DataArray)
    assert isinstance(pvals[1], xr.DataArray)


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
])
def test_heterogeneous_patterns(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    patterns, pvals = mca_model.heterogeneous_patterns()
    assert isinstance(patterns[0], xr.DataArray)
    assert isinstance(patterns[1], xr.DataArray)
    assert isinstance(pvals[0], xr.DataArray)
    assert isinstance(pvals[1], xr.DataArray)


@pytest.mark.parametrize('dim', [
    (('time',)),
    (('lat', 'lon')),
    (('lon', 'lat')),
])
def test_compute(mca_model, mock_dask_data_array, dim):
    mca_model.fit(mock_dask_data_array, mock_dask_data_array, (dim))
    assert isinstance(mca_model._singular_values.data, DaskArray)
    assert isinstance(mca_model._explained_variance.data, DaskArray)
    mca_model.compute()
    assert isinstance(mca_model._singular_values.data, np.ndarray)
    assert isinstance(mca_model._explained_variance.data, np.ndarray)





