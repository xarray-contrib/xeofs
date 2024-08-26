import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xeofs.models.cpcca import ContinuousPowerCCA


def generate_random_data(shape, lazy=False):
    rng = np.random.default_rng(142)
    if lazy:
        return xr.DataArray(
            da.random.random(shape, chunks=(5, 5)),
            dims=["sample", "feature"],
            coords={"sample": np.arange(shape[0]), "feature": np.arange(shape[1])},
        )
    else:
        return xr.DataArray(
            rng.random(shape),
            dims=["sample", "feature"],
            coords={"sample": np.arange(shape[0]), "feature": np.arange(shape[1])},
        )


def generate_well_conditioned_data(lazy=False):
    rng = np.random.default_rng(142)
    t = np.linspace(0, 50, 200)
    std = 0.1
    x1 = np.sin(t)[:, None] + rng.normal(0, std, size=(200, 2))
    x2 = np.sin(t)[:, None] + rng.normal(0, std, size=(200, 3))
    x1[:, 1] = x1[:, 1] ** 2
    x2[:, 1] = x2[:, 1] ** 3
    x2[:, 2] = abs(x2[:, 2]) ** (0.5)
    coords_time = np.arange(len(t))
    coords_fx = [1, 2]
    coords_fy = [1, 2, 3]
    X = xr.DataArray(
        x1,
        dims=["sample", "feature"],
        coords={"sample": coords_time, "feature": coords_fx},
    )
    Y = xr.DataArray(
        x2,
        dims=["sample", "feature"],
        coords={"sample": coords_time, "feature": coords_fy},
    )
    if lazy:
        X = X.chunk({"sample": 5, "feature": -1})
        Y = Y.chunk({"sample": 5, "feature": -1})
        return X, Y
    else:
        return X, Y


@pytest.fixture
def cpcca():
    return ContinuousPowerCCA(n_modes=1)


def test_initialization():
    model = ContinuousPowerCCA()
    assert model is not None


def test_fit(cpcca):
    X, Y = generate_well_conditioned_data()
    cpcca.fit(X, Y, dim="sample")
    assert hasattr(cpcca, "preprocessor1")
    assert hasattr(cpcca, "preprocessor2")
    assert hasattr(cpcca, "data")


def test_fit_empty_data(cpcca):
    with pytest.raises(ValueError):
        cpcca.fit(xr.DataArray(), xr.DataArray(), "time")


def test_fit_invalid_dims(cpcca):
    X, Y = generate_well_conditioned_data()
    with pytest.raises(ValueError):
        cpcca.fit(X, Y, dim=("invalid_dim1", "invalid_dim2"))


def test_transform(cpcca):
    X, Y = generate_well_conditioned_data()
    cpcca.fit(X, Y, dim="sample")
    result = cpcca.transform(X, Y)
    assert isinstance(result, list)
    assert isinstance(result[0], xr.DataArray)


def test_transform_unseen_data(cpcca):
    X, Y = generate_well_conditioned_data()
    x = X.isel(sample=slice(151, 200))
    y = Y.isel(sample=slice(151, 200))
    X = X.isel(sample=slice(None, 150))
    Y = Y.isel(sample=slice(None, 150))

    cpcca.fit(X, Y, "sample")
    result = cpcca.transform(x, y)
    assert isinstance(result, list)
    assert isinstance(result[0], xr.DataArray)
    # Check that unseen data can be transformed
    assert result[0].notnull().all()
    assert result[1].notnull().all()


def test_inverse_transform(cpcca):
    X, Y = generate_well_conditioned_data()
    cpcca.fit(X, Y, "sample")
    # Assuming mode as 1 for simplicity
    scores1 = cpcca.data["scores1"].sel(mode=1)
    scores2 = cpcca.data["scores2"].sel(mode=1)
    Xrec1, Xrec2 = cpcca.inverse_transform(scores1, scores2)
    assert isinstance(Xrec1, xr.DataArray)
    assert isinstance(Xrec2, xr.DataArray)


def test_squared_covariance_fraction(cpcca):
    X, Y = generate_well_conditioned_data()
    cpcca.fit(X, Y, "sample")
    scf = cpcca.squared_covariance_fraction()
    assert isinstance(scf, xr.DataArray)
    assert scf.sum("mode") <= 1.00001, "Squared covariance fraction is greater than 1"


@pytest.mark.parametrize(
    "alpha,use_pca",
    [
        (1.0, False),
        (0.5, False),
        (0.0, False),
        (1.0, True),
        (0.5, True),
        (0.0, True),
    ],
)
def test_total_squared_covariance(alpha, use_pca):
    X, Y = generate_well_conditioned_data()

    # Compute total squared covariance
    X_ = X.rename({"feature": "x"})
    Y_ = Y.rename({"feature": "y"})
    cov_mat = xr.cov(X_, Y_, dim="sample")
    tsc = (cov_mat**2).sum()

    cpcca = ContinuousPowerCCA(
        n_modes=2, alpha=alpha, use_pca=use_pca, n_pca_modes="all"
    )
    cpcca.fit(X, Y, "sample")
    tsc_model = cpcca.data["total_squared_covariance"]
    xr.testing.assert_allclose(tsc, tsc_model)


def test_alpha_integer():
    X, Y = generate_well_conditioned_data()

    cpcca = ContinuousPowerCCA(n_modes=2, alpha=1, use_pca=False)
    cpcca.fit(X, Y, "sample")


def test_fit_different_coordinates():
    """Like a lagged CCA scenario"""
    X, Y = generate_well_conditioned_data()
    X = X.isel(sample=slice(0, 99))
    Y = Y.isel(sample=slice(100, 199))
    cpcca = ContinuousPowerCCA(n_modes=2, alpha=1, use_pca=False)
    cpcca.fit(X, Y, "sample")
    r = cpcca.cross_correlation_coefficients()
    # Correlation coefficents are not zero
    assert np.all(r > np.finfo(r.dtype).eps)


@pytest.mark.parametrize(
    "dim",
    [
        (("time",)),
        (("lat", "lon")),
        (("lon", "lat")),
    ],
)
def test_components(mock_data_array, dim):
    cpcca = ContinuousPowerCCA(n_modes=2, alpha=1, use_pca=False)
    cpcca.fit(mock_data_array, mock_data_array, dim)
    components1, components2 = cpcca.components()
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
    "shapeX,shapeY,alpha,use_pca",
    [
        ((20, 10), (20, 10), 1.0, False),
        ((20, 40), (20, 30), 1.0, False),
        ((20, 10), (20, 40), 1.0, False),
        ((20, 10), (20, 10), 0.5, False),
        ((20, 40), (20, 30), 0.5, False),
        ((20, 10), (20, 40), 0.5, False),
        ((20, 10), (20, 10), 1.0, True),
        ((20, 40), (20, 30), 1.0, True),
        ((20, 10), (20, 40), 1.0, True),
        ((20, 10), (20, 10), 0.5, True),
        ((20, 40), (20, 30), 0.5, True),
        ((20, 10), (20, 40), 0.5, True),
    ],
)
def test_components_coordinates(shapeX, shapeY, alpha, use_pca):
    # Test that the components have the right coordinates
    X = generate_random_data(shapeX)
    Y = generate_random_data(shapeY)

    cpcca = ContinuousPowerCCA(
        n_modes=2, alpha=alpha, use_pca=use_pca, n_pca_modes="all"
    )
    cpcca.fit(X, Y, "sample")
    components1, components2 = cpcca.components()
    xr.testing.assert_equal(components1.coords["feature"], X.coords["feature"])
    xr.testing.assert_equal(components2.coords["feature"], Y.coords["feature"])


# @pytest.mark.parametrize(
#     "dim",
#     [
#         (("time",)),
#         (("lat", "lon")),
#         (("lon", "lat")),
#     ],
# )
# def test_singular_values(mca_model, mock_data_array, dim):
#     mca_model.fit(mock_data_array, mock_data_array, dim)
#     n_modes = mca_model.get_params()["n_modes"]
#     svals = mca_model.singular_values()
#     assert isinstance(svals, xr.DataArray)
#     assert svals.size == n_modes


# @pytest.mark.parametrize(
#     "dim",
#     [
#         (("time",)),
#         (("lat", "lon")),
#         (("lon", "lat")),
#     ],
# )
# def test_covariance_fraction(mca_model, mock_data_array, dim):
#     mca_model.fit(mock_data_array, mock_data_array, dim)
#     cf = mca_model.covariance_fraction()
#     assert isinstance(cf, xr.DataArray)
#     assert cf.sum("mode") <= 1.00001, "Covariance fraction is greater than 1"


# @pytest.mark.parametrize(
#     "dim",
#     [
#         (("time",)),
#         (("lat", "lon")),
#         (("lon", "lat")),
#     ],
# )
# def test_components_dataset(mca_model, mock_dataset, dim):
#     mca_model.fit(mock_dataset, mock_dataset, dim)
#     components1, components2 = mca_model.components()
#     feature_dims = tuple(set(mock_dataset.dims) - set(dim))
#     assert isinstance(components1, xr.Dataset)
#     assert isinstance(components2, xr.Dataset)
#     assert set(components1.data_vars) == set(
#         mock_dataset.data_vars
#     ), "Components does not have the same data variables as the input Dataset"
#     assert set(components2.data_vars) == set(
#         mock_dataset.data_vars
#     ), "Components does not have the same data variables as the input Dataset"
#     assert set(components1.dims) == set(
#         ("mode",) + feature_dims
#     ), "Components does not have the right feature dimensions"
#     assert set(components2.dims) == set(
#         ("mode",) + feature_dims
#     ), "Components does not have the right feature dimensions"


# @pytest.mark.parametrize(
#     "dim",
#     [
#         (("time",)),
#         (("lat", "lon")),
#         (("lon", "lat")),
#     ],
# )
# def test_components_dataarray_list(mca_model, mock_data_array_list, dim):
#     mca_model.fit(mock_data_array_list, mock_data_array_list, dim)
#     components1, components2 = mca_model.components()
#     feature_dims = [tuple(set(data.dims) - set(dim)) for data in mock_data_array_list]
#     assert isinstance(components1, list)
#     assert isinstance(components2, list)
#     assert len(components1) == len(mock_data_array_list)
#     assert len(components2) == len(mock_data_array_list)
#     assert isinstance(components1[0], xr.DataArray)
#     assert isinstance(components2[0], xr.DataArray)
#     for comp, feat_dims in zip(components1, feature_dims):
#         assert set(comp.dims) == set(
#             ("mode",) + feat_dims
#         ), "Components1 does not have the right feature dimensions"
#     for comp, feat_dims in zip(components2, feature_dims):
#         assert set(comp.dims) == set(
#             ("mode",) + feat_dims
#         ), "Components2 does not have the right feature dimensions"


# @pytest.mark.parametrize(
#     "dim",
#     [
#         (("time",)),
#         (("lat", "lon")),
#         (("lon", "lat")),
#     ],
# )
# def test_scores(mca_model, mock_data_array, dim):
#     mca_model.fit(mock_data_array, mock_data_array, dim)
#     scores1, scores2 = mca_model.scores()
#     assert isinstance(scores1, xr.DataArray)
#     assert isinstance(scores2, xr.DataArray)
#     assert set(scores1.dims) == set(
#         (dim + ("mode",))
#     ), "Scores1 does not have the right dimensions"
#     assert set(scores2.dims) == set(
#         (dim + ("mode",))
#     ), "Scores2 does not have the right dimensions"


# @pytest.mark.parametrize(
#     "dim",
#     [
#         (("time",)),
#         (("lat", "lon")),
#         (("lon", "lat")),
#     ],
# )
# def test_scores_dataset(mca_model, mock_dataset, dim):
#     mca_model.fit(mock_dataset, mock_dataset, dim)
#     scores1, scores2 = mca_model.scores()
#     assert isinstance(scores1, xr.DataArray)
#     assert isinstance(scores2, xr.DataArray)
#     assert set(scores1.dims) == set(
#         (dim + ("mode",))
#     ), "Scores1 does not have the right dimensions"
#     assert set(scores2.dims) == set(
#         (dim + ("mode",))
#     ), "Scores2 does not have the right dimensions"


# @pytest.mark.parametrize(
#     "dim",
#     [
#         (("time",)),
#         (("lat", "lon")),
#         (("lon", "lat")),
#     ],
# )
# def test_scores_dataarray_list(mca_model, mock_data_array_list, dim):
#     mca_model.fit(mock_data_array_list, mock_data_array_list, dim)
#     scores1, scores2 = mca_model.scores()
#     assert isinstance(scores1, xr.DataArray)
#     assert isinstance(scores2, xr.DataArray)
#     assert set(scores1.dims) == set(
#         (dim + ("mode",))
#     ), "Scores1 does not have the right dimensions"
#     assert set(scores2.dims) == set(
#         (dim + ("mode",))
#     ), "Scores2 does not have the right dimensions"


# @pytest.mark.parametrize(
#     "dim",
#     [
#         (("time",)),
#         (("lat", "lon")),
#         (("lon", "lat")),
#     ],
# )
# def test_homogeneous_patterns(mca_model, mock_data_array, dim):
#     mca_model.fit(mock_data_array, mock_data_array, dim)
#     patterns, pvals = mca_model.homogeneous_patterns()
#     assert isinstance(patterns[0], xr.DataArray)
#     assert isinstance(patterns[1], xr.DataArray)
#     assert isinstance(pvals[0], xr.DataArray)
#     assert isinstance(pvals[1], xr.DataArray)


# @pytest.mark.parametrize(
#     "dim",
#     [
#         (("time",)),
#         (("lat", "lon")),
#         (("lon", "lat")),
#     ],
# )
# def test_heterogeneous_patterns(mca_model, mock_data_array, dim):
#     mca_model.fit(mock_data_array, mock_data_array, dim)
#     patterns, pvals = mca_model.heterogeneous_patterns()
#     assert isinstance(patterns[0], xr.DataArray)
#     assert isinstance(patterns[1], xr.DataArray)
#     assert isinstance(pvals[0], xr.DataArray)
#     assert isinstance(pvals[1], xr.DataArray)


# @pytest.mark.parametrize(
#     "dim, compute",
#     [
#         (("time",), True),
#         (("lat", "lon"), True),
#         (("lon", "lat"), True),
#         (("time",), False),
#         (("lat", "lon"), False),
#         (("lon", "lat"), False),
#     ],
# )
# def test_compute(mock_dask_data_array, dim, compute):
#     mca_model = MCA(n_modes=10, compute=compute, n_pca_modes=10)
#     mca_model.fit(mock_dask_data_array, mock_dask_data_array, (dim))

#     if compute:
#         assert not data_is_dask(mca_model.data["squared_covariance"])
#         assert not data_is_dask(mca_model.data["components1"])
#         assert not data_is_dask(mca_model.data["components2"])

#     else:
#         assert data_is_dask(mca_model.data["squared_covariance"])
#         assert data_is_dask(mca_model.data["components1"])
#         assert data_is_dask(mca_model.data["components2"])


# @pytest.mark.parametrize(
#     "dim",
#     [
#         (("time",)),
#         (("lat", "lon")),
#         (("lon", "lat")),
#     ],
# )
# @pytest.mark.parametrize("engine", ["netcdf4", "zarr"])
# def test_save_load(dim, mock_data_array, tmp_path, engine):
#     """Test save/load methods in MCA class, ensuring that we can
#     roundtrip the model and get the same results when transforming
#     data."""
#     original = MCA()
#     original.fit(mock_data_array, mock_data_array, dim)

#     # Save the EOF model
#     original.save(tmp_path / "mca", engine=engine)

#     # Check that the EOF model has been saved
#     assert (tmp_path / "mca").exists()

#     # Recreate the model from saved file
#     loaded = MCA.load(tmp_path / "mca", engine=engine)

#     # Check that the params and DataContainer objects match
#     assert original.get_params() == loaded.get_params()
#     assert all([key in loaded.data for key in original.data])
#     for key in original.data:
#         if original.data._allow_compute[key]:
#             assert loaded.data[key].equals(original.data[key])
#         else:
#             # but ensure that input data is not saved by default
#             assert loaded.data[key].size <= 1
#             assert loaded.data[key].attrs["placeholder"] is True

#     # Test that the recreated model can be used to transform new data
#     assert np.allclose(
#         original.transform(mock_data_array, mock_data_array),
#         loaded.transform(mock_data_array, mock_data_array),
#     )

#     # The loaded model should also be able to inverse_transform new data
#     assert np.allclose(
#         original.inverse_transform(*original.scores()),
#         loaded.inverse_transform(*loaded.scores()),
#     )


# @pytest.mark.parametrize(
#     "dim",
#     [
#         (("time",)),
#         (("lat", "lon")),
#         (("lon", "lat")),
#     ],
# )
# def test_serialize_deserialize_dataarray(dim, mock_data_array):
#     """Test roundtrip serialization when the model is fit on a DataArray."""
#     model = MCA()
#     model.fit(mock_data_array, mock_data_array, dim)
#     dt = model.serialize()
#     rebuilt_model = MCA.deserialize(dt)
#     assert np.allclose(
#         model.transform(mock_data_array), rebuilt_model.transform(mock_data_array)
#     )


# @pytest.mark.parametrize(
#     "dim",
#     [
#         (("time",)),
#         (("lat", "lon")),
#         (("lon", "lat")),
#     ],
# )
# def test_serialize_deserialize_dataset(dim, mock_dataset):
#     """Test roundtrip serialization when the model is fit on a Dataset."""
#     model = MCA()
#     model.fit(mock_dataset, mock_dataset, dim)
#     dt = model.serialize()
#     rebuilt_model = MCA.deserialize(dt)
#     assert np.allclose(
#         model.transform(mock_dataset), rebuilt_model.transform(mock_dataset)
#     )
