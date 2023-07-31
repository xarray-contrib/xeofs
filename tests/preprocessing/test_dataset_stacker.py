import pytest
import xarray as xr
import numpy as np

from xeofs.preprocessing.stacker import SingleDatasetStacker


@pytest.mark.parametrize('dim_sample, dim_feature', [
    (('time',), ('lat', 'lon')),
    (('time',), ('lon', 'lat')),
    (('lat', 'lon'), ('time',)),
    (('lon', 'lat'), ('time',)),
    ])
def test_DatasetStacker_fit_transform(mock_dataset, dim_sample, dim_feature):
    stacker = SingleDatasetStacker()
    stacked = stacker.fit_transform(mock_dataset, dim_sample, dim_feature)

    # check output type and dimensions
    assert isinstance(stacked, xr.DataArray)
    assert set(stacked.dims) == {'sample', 'feature'}

    # check if all NaN rows or columns have been dropped
    assert not stacked.isnull().any()

    # check the size of the output data
    assert stacked.size > 0


@pytest.mark.parametrize('dim_sample, dim_feature', [
    (('time',), ('lat', 'lon')),
    (('time',), ('lon', 'lat')),
    (('lat', 'lon'), ('time',)),
    (('lon', 'lat'), ('time',)),
    ])
def test_DatasetStacker_transform(mock_dataset, dim_sample, dim_feature):
    stacker = SingleDatasetStacker()
    stacker.fit_transform(mock_dataset, dim_sample, dim_feature)

    # create a new dataset for testing the transform function
    new_data = mock_dataset.copy()
    transformed = stacker.transform(new_data)

    assert isinstance(transformed, xr.DataArray)
    assert set(transformed.dims) == {'sample', 'feature'}
    assert not transformed.isnull().any()
    assert transformed.size > 0


@pytest.mark.parametrize('dim_sample, dim_feature', [
    (('time',), ('lat', 'lon')),
    (('time',), ('lon', 'lat')),
    (('lat', 'lon'), ('time',)),
    (('lon', 'lat'), ('time',)),
    ])
def test_DatasetStacker_inverse_transform_data(mock_dataset, dim_sample, dim_feature):
    stacker = SingleDatasetStacker()
    stacked = stacker.fit_transform(mock_dataset, dim_sample, dim_feature)

    inverse_transformed = stacker.inverse_transform_data(stacked)
    assert isinstance(inverse_transformed, xr.Dataset)

    for var in inverse_transformed.data_vars:
        xr.testing.assert_equal(inverse_transformed[var], mock_dataset[var])


@pytest.mark.parametrize('dim_sample, dim_feature', [
    (('time',), ('lat', 'lon')),
    (('time',), ('lon', 'lat')),
    (('lat', 'lon'), ('time',)),
    (('lon', 'lat'), ('time',)),
    ])
def test_DatasetStacker_inverse_transform_components(mock_dataset, dim_sample, dim_feature):
    stacker = SingleDatasetStacker()
    stacked = stacker.fit_transform(mock_dataset, dim_sample, dim_feature)

    # dummy components
    components = xr.DataArray(
        np.random.normal(size=(len(stacker.coords_out_['feature']), 10)),
        dims=('feature', 'mode'),
        coords={'feature': stacker.coords_out_['feature']}
    )
    inverse_transformed = stacker.inverse_transform_components(components)

    # check output type and dimensions
    assert isinstance(inverse_transformed, xr.Dataset)
    assert set(inverse_transformed.dims) == set(dim_feature + ('mode',))

    assert set(mock_dataset.data_vars) == set(inverse_transformed.data_vars), 'Dataset variables are not the same.'


@pytest.mark.parametrize('dim_sample, dim_feature', [
    (('time',), ('lat', 'lon')),
    (('time',), ('lon', 'lat')),
    (('lat', 'lon'), ('time',)),
    (('lon', 'lat'), ('time',)),
    ])
def test_DatasetStacker_inverse_transform_scores(mock_dataset, dim_sample, dim_feature):
    stacker = SingleDatasetStacker()
    stacked = stacker.fit_transform(mock_dataset, dim_sample, dim_feature)

    # dummy scores
    scores = xr.DataArray(
        np.random.rand(len(stacker.coords_out_['sample']), 10),
        dims=('sample', 'mode'),
        coords={'sample': stacker.coords_out_['sample']}
    )
    inverse_transformed = stacker.inverse_transform_scores(scores)
    
    assert isinstance(inverse_transformed, xr.DataArray)
    assert set(inverse_transformed.dims) == set(dim_sample + ('mode',))
    
    # check that sample coordinates are preserved
    for dim, coords in mock_dataset.coords.items():
        if dim in dim_sample:
            assert inverse_transformed.coords[dim].size == coords.size, 'Dimension {} has different size.'.format(dim)
            


@pytest.mark.parametrize('dim_sample, dim_feature', [
    (('time',), ('lat', 'lon')),
    (('time',), ('lon', 'lat')),
    (('lat', 'lon'), ('time',)),
    (('lon', 'lat'), ('time',)),
    ])
def test_DatasetStacker_fit_transform_raises_on_invalid_dims(mock_dataset, dim_sample, dim_feature):
    stacker = SingleDatasetStacker()
    with pytest.raises(ValueError):
        stacker.fit_transform(mock_dataset, ('invalid_dim',), dim_feature)


def test_DatasetStacker_fit_transform_raises_on_isolated_nans(mock_data_array_isolated_nans):
    stacker = SingleDatasetStacker()
    invalid_dataset = xr.Dataset({'var': mock_data_array_isolated_nans})
    with pytest.raises(ValueError):
        stacker.fit_transform(invalid_dataset, ('time',), ('lat', 'lon'))


@pytest.mark.parametrize('dim_sample, dim_feature', [
    (('time',), ('lat', 'lon')),
    (('time',), ('lon', 'lat')),
    (('lat', 'lon'), ('time',)),
    (('lon', 'lat'), ('time',)),
    ])
def test_DatasetStacker_fit_transform_passes_on_full_dimensional_nans(mock_data_array_full_dimensional_nans, dim_sample, dim_feature):
    stacker = SingleDatasetStacker()
    valid_dataset = xr.Dataset({'var': mock_data_array_full_dimensional_nans})
    try:
        stacker.fit_transform(valid_dataset, dim_sample, dim_feature)
    except ValueError:
        pytest.fail("fit_transform() raised ValueError unexpectedly!")