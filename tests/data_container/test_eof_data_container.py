import pytest
import numpy as np
import xarray as xr
from dask.array import Array as DaskArray  # type: ignore


from xeofs.data_container.eof_data_container import EOFDataContainer


@pytest.fixture
def components():
    return xr.DataArray(
        np.random.rand(10, 10),
        dims=('feature', 'mode'),
        coords={'feature': np.arange(10), 'mode': np.arange(10)},
        name='components',
    )

@pytest.fixture
def scores():
    return xr.DataArray(
        np.random.rand(100, 10),
        dims=('sample', 'mode'),
        coords={'sample': np.arange(100), 'mode': np.arange(10)},
        name='scores',
    )

@pytest.fixture
def explained_variance():
    return xr.DataArray(
        np.random.rand(10),
        dims=('mode',),
        coords={'mode': np.arange(10)},
        name='explained_variance'
    )

def test_eof_results_init(mock_data_array, components, scores, explained_variance):
    data = mock_data_array.stack(sample=('time',), feature=('lat', 'lon'))
    data = data - data.mean('sample')
    total_variance = explained_variance.sum()
    idx_modes_sorted = explained_variance.argsort()[::-1]
    results = EOFDataContainer(
        input_data=data,
        components=components,
        scores=scores,
        explained_variance=explained_variance,
        total_variance=total_variance,
        idx_modes_sorted=idx_modes_sorted
    )
    assert results.input_data is data
    assert results.components is components
    assert results.scores is scores
    assert results.explained_variance is explained_variance
    assert results.total_variance is total_variance
    assert results.idx_modes_sorted is idx_modes_sorted


def test_eof_results_init_invalid(mock_data_array, components, scores, explained_variance):
    data = mock_data_array.stack(sample=('time',), feature=('lat', 'lon'))
    data = data - data.mean('sample')
    total_variance = explained_variance.sum()
    idx_modes_sorted = explained_variance.argsort()[::-1]
    with pytest.raises(ValueError):
        EOFDataContainer(
            input_data=data.rename({'feature': 'lon'}),
            components=components,
            scores=scores,
            explained_variance=explained_variance,
            total_variance=total_variance,
            idx_modes_sorted=idx_modes_sorted
        )
    with pytest.raises(ValueError):
        EOFDataContainer(
            input_data=data,
            components=components.rename({'feature': 'lon'}),
            scores=scores,
            explained_variance=explained_variance,
            total_variance=total_variance,
            idx_modes_sorted=idx_modes_sorted
        )
    with pytest.raises(ValueError):
        EOFDataContainer(
            input_data=data,
            components=components,
            scores=scores.rename({'sample': 'date'}),
            explained_variance=explained_variance,
            total_variance=total_variance,
            idx_modes_sorted=idx_modes_sorted
        )
    with pytest.raises(ValueError):
        EOFDataContainer(
            input_data=data,
            components=components,
            scores=scores,
            explained_variance=explained_variance.rename({'mode': 'number'}),
            total_variance=total_variance,
            idx_modes_sorted=idx_modes_sorted
        )
    
def test_eof_results_compute(mock_dask_data_array, components, scores, explained_variance):
    '''Check that dask arrays are computed correctly.'''
    data = mock_dask_data_array.stack(sample=('time',), feature=('lat', 'lon'))
    data = data - data.mean('sample')
    total_variance = explained_variance.chunk({'mode':2}).sum()
    idx_modes_sorted = explained_variance.argsort()[::-1]
    results = EOFDataContainer(
        input_data=data.chunk({'sample': 2}),
        components=components.chunk({'feature': 2}),
        scores=scores.chunk({'sample': 2}),
        explained_variance=explained_variance.chunk({'mode': 2}),
        total_variance=total_variance,
        idx_modes_sorted=idx_modes_sorted
    )
    # Check that the components and scores are dask arrays
    assert isinstance(results.input_data.data, DaskArray)
    assert isinstance(results.components.data, DaskArray)
    assert isinstance(results.scores.data, DaskArray)
    assert isinstance(results.explained_variance.data, DaskArray)
    assert isinstance(results.total_variance.data, DaskArray)

    # Check that the components and scores are computed correctly
    results.compute()
    assert isinstance(results.input_data.data, DaskArray), 'input_data should still be a dask array'
    assert isinstance(results.components.data, np.ndarray)
    assert isinstance(results.scores.data, np.ndarray)
    assert isinstance(results.explained_variance.data, np.ndarray)
    assert isinstance(results.total_variance.data, np.ndarray)
