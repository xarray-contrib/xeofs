import numpy as np
import pytest
import xarray as xr

from xeofs.cross import MCA

from ...utilities import data_is_dask


@pytest.fixture
def mca_model():
    return MCA()


def test_initialization():
    mca = MCA()
    assert mca is not None


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_fit(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    assert hasattr(mca_model, "preprocessor1")
    assert hasattr(mca_model, "preprocessor2")
    assert hasattr(mca_model, "data")


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_fit_empty_data(mca_model, dim):
    with pytest.raises(ValueError):
        mca_model.fit(xr.DataArray(), xr.DataArray(), dim)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_fit_invalid_dims(mca_model, mock_data_array, dim):
    with pytest.raises(ValueError):
        mca_model.fit(
            mock_data_array, mock_data_array, dim=("invalid_dim1", "invalid_dim2")
        )


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_fit_with_dataset(mca_model, mock_dataset, dim):
    mca_model.fit(mock_dataset, mock_dataset, dim)
    assert hasattr(mca_model, "preprocessor1")
    assert hasattr(mca_model, "preprocessor2")
    assert hasattr(mca_model, "data")


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_fit_with_dataarray_list(mca_model, mock_data_array_list, dim):
    mca_model.fit(mock_data_array_list, mock_data_array_list, dim)
    assert hasattr(mca_model, "preprocessor1")
    assert hasattr(mca_model, "preprocessor2")
    assert hasattr(mca_model, "data")


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_transform(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    result = mca_model.transform(X=mock_data_array, Y=mock_data_array)
    assert isinstance(result, list)
    assert isinstance(result[0], xr.DataArray)


@pytest.mark.parametrize("dim", [(("time",))])
def test_transform_unseen_data(mca_model, mock_data_array, dim):
    data = mock_data_array.isel(time=slice(0, 20))
    data_unseen = mock_data_array.isel(time=slice(21, None))

    mca_model.fit(data, data, dim)
    result = mca_model.transform(X=data_unseen, Y=data_unseen)
    assert isinstance(result, list)
    assert isinstance(result[0], xr.DataArray)
    # Check that unseen data can be transformed
    assert result[0].notnull().all()
    assert result[1].notnull().all()


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_inverse_transform(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    # Assuming mode as 1 for simplicity
    scores1 = mca_model.data["scores1"].isel(mode=1)
    scores2 = mca_model.data["scores2"].isel(mode=1)
    Xrec1, Xrec2 = mca_model.inverse_transform(scores1, scores2)
    assert isinstance(Xrec1, xr.DataArray)
    assert isinstance(Xrec2, xr.DataArray)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_squared_covariance_fraction(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    scf = mca_model.squared_covariance_fraction()
    assert isinstance(scf, xr.DataArray)
    assert all(scf <= 1), "Squared covariance fraction is greater than 1"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_covariance_fraction(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    cf = mca_model.covariance_fraction_CD95()
    assert isinstance(cf, xr.DataArray)
    assert all(cf <= 1), "Squared covariance fraction is greater than 1"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    components1, components2 = mca_model.components()
    feature_dims = tuple(set(mock_data_array.dims) - set(dim))
    assert isinstance(components1, xr.DataArray)
    assert isinstance(components2, xr.DataArray)
    assert set(components1.dims) == set(
        ("mode",) + feature_dims
    ), "Components1 does not have the right feature dimensions"
    assert set(components2.dims) == set(
        ("mode",) + feature_dims
    ), "Components2 does not have the right feature dimensions"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components_dataset(mca_model, mock_dataset, dim):
    mca_model.fit(mock_dataset, mock_dataset, dim)
    components1, components2 = mca_model.components()
    feature_dims = tuple(set(mock_dataset.dims) - set(dim))
    assert isinstance(components1, xr.Dataset)
    assert isinstance(components2, xr.Dataset)
    assert set(components1.data_vars) == set(
        mock_dataset.data_vars
    ), "Components does not have the same data variables as the input Dataset"
    assert set(components2.data_vars) == set(
        mock_dataset.data_vars
    ), "Components does not have the same data variables as the input Dataset"
    assert set(components1.dims) == set(
        ("mode",) + feature_dims
    ), "Components does not have the right feature dimensions"
    assert set(components2.dims) == set(
        ("mode",) + feature_dims
    ), "Components does not have the right feature dimensions"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components_dataarray_list(mca_model, mock_data_array_list, dim):
    mca_model.fit(mock_data_array_list, mock_data_array_list, dim)
    components1, components2 = mca_model.components()
    feature_dims = [tuple(set(data.dims) - set(dim)) for data in mock_data_array_list]
    assert isinstance(components1, list)
    assert isinstance(components2, list)
    assert len(components1) == len(mock_data_array_list)
    assert len(components2) == len(mock_data_array_list)
    assert isinstance(components1[0], xr.DataArray)
    assert isinstance(components2[0], xr.DataArray)
    for comp, feat_dims in zip(components1, feature_dims):
        assert set(comp.dims) == set(
            ("mode",) + feat_dims
        ), "Components1 does not have the right feature dimensions"
    for comp, feat_dims in zip(components2, feature_dims):
        assert set(comp.dims) == set(
            ("mode",) + feat_dims
        ), "Components2 does not have the right feature dimensions"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_scores(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    scores1, scores2 = mca_model.scores()
    assert isinstance(scores1, xr.DataArray)
    assert isinstance(scores2, xr.DataArray)
    assert set(scores1.dims) == set(
        (dim + ("mode",))
    ), "Scores1 does not have the right dimensions"
    assert set(scores2.dims) == set(
        (dim + ("mode",))
    ), "Scores2 does not have the right dimensions"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_scores_dataset(mca_model, mock_dataset, dim):
    mca_model.fit(mock_dataset, mock_dataset, dim)
    scores1, scores2 = mca_model.scores()
    assert isinstance(scores1, xr.DataArray)
    assert isinstance(scores2, xr.DataArray)
    assert set(scores1.dims) == set(
        (dim + ("mode",))
    ), "Scores1 does not have the right dimensions"
    assert set(scores2.dims) == set(
        (dim + ("mode",))
    ), "Scores2 does not have the right dimensions"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_scores_dataarray_list(mca_model, mock_data_array_list, dim):
    mca_model.fit(mock_data_array_list, mock_data_array_list, dim)
    scores1, scores2 = mca_model.scores()
    assert isinstance(scores1, xr.DataArray)
    assert isinstance(scores2, xr.DataArray)
    assert set(scores1.dims) == set(
        (dim + ("mode",))
    ), "Scores1 does not have the right dimensions"
    assert set(scores2.dims) == set(
        (dim + ("mode",))
    ), "Scores2 does not have the right dimensions"


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_homogeneous_patterns(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    patterns, pvals = mca_model.homogeneous_patterns()
    assert isinstance(patterns[0], xr.DataArray)
    assert isinstance(patterns[1], xr.DataArray)
    assert isinstance(pvals[0], xr.DataArray)
    assert isinstance(pvals[1], xr.DataArray)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_heterogeneous_patterns(mca_model, mock_data_array, dim):
    mca_model.fit(mock_data_array, mock_data_array, dim)
    patterns, pvals = mca_model.heterogeneous_patterns()
    assert isinstance(patterns[0], xr.DataArray)
    assert isinstance(patterns[1], xr.DataArray)
    assert isinstance(pvals[0], xr.DataArray)
    assert isinstance(pvals[1], xr.DataArray)


@pytest.mark.parametrize(
    "dim, compute",
    [
        (("time",), True),
        (("lat", "lon"), True),
        (("lon", "lat"), True),
        (("time",), False),
        (("lat", "lon"), False),
        (("lon", "lat"), False),
    ],
)
def test_compute(mock_dask_data_array, dim, compute):
    mca_model = MCA(n_modes=10, compute=compute, n_pca_modes=10)
    mca_model.fit(mock_dask_data_array, mock_dask_data_array, (dim))

    if compute:
        assert not data_is_dask(mca_model.data["singular_values"])
        assert not data_is_dask(mca_model.data["components1"])
        assert not data_is_dask(mca_model.data["components2"])

    else:
        assert data_is_dask(mca_model.data["singular_values"])
        assert data_is_dask(mca_model.data["components1"])
        assert data_is_dask(mca_model.data["components2"])


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
@pytest.mark.parametrize("engine", ["netcdf4", "zarr"])
def test_save_load(dim, mock_data_array, tmp_path, engine):
    """Test save/load methods in MCA class, ensuring that we can
    roundtrip the model and get the same results when transforming
    data."""
    original = MCA()
    original.fit(mock_data_array, mock_data_array, dim)

    # Save the EOF model
    original.save(tmp_path / "mca", engine=engine)

    # Check that the EOF model has been saved
    assert (tmp_path / "mca").exists()

    # Recreate the model from saved file
    loaded = MCA.load(tmp_path / "mca", engine=engine)

    # Check that the params and DataContainer objects match
    assert original.get_params() == loaded.get_params()
    assert all([key in loaded.data for key in original.data])
    for key in original.data:
        if original.data._allow_compute[key]:
            assert loaded.data[key].equals(original.data[key])
        else:
            # but ensure that input data is not saved by default
            assert loaded.data[key].size <= 1
            assert loaded.data[key].attrs["placeholder"] is True

    # Test that the recreated model can be used to transform new data
    assert np.allclose(
        original.transform(mock_data_array, mock_data_array),
        loaded.transform(mock_data_array, mock_data_array),
    )

    # The loaded model should also be able to inverse_transform new data
    assert np.allclose(
        original.inverse_transform(*original.scores()),
        loaded.inverse_transform(*loaded.scores()),
    )


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_serialize_deserialize_dataarray(dim, mock_data_array):
    """Test roundtrip serialization when the model is fit on a DataArray."""
    model = MCA()
    model.fit(mock_data_array, mock_data_array, dim)
    dt = model.serialize()
    rebuilt_model = MCA.deserialize(dt)
    assert np.allclose(
        model.transform(mock_data_array), rebuilt_model.transform(mock_data_array)
    )


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_serialize_deserialize_dataset(dim, mock_dataset):
    """Test roundtrip serialization when the model is fit on a Dataset."""
    model = MCA()
    model.fit(mock_dataset, mock_dataset, dim)
    dt = model.serialize()
    rebuilt_model = MCA.deserialize(dt)
    assert np.allclose(
        model.transform(mock_dataset), rebuilt_model.transform(mock_dataset)
    )
