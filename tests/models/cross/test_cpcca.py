import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xeofs.cross import CPCCA


def generate_random_data(shape, lazy=False, seed=142):
    rng = np.random.default_rng(seed)
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
    return CPCCA(n_modes=1)


def test_initialization():
    model = CPCCA()
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
def test_squared_covariance_fraction(alpha, use_pca):
    X, Y = generate_well_conditioned_data()
    cpcca = CPCCA(n_modes=2, alpha=alpha, use_pca=use_pca, n_pca_modes="all")
    cpcca.fit(X, Y, "sample")
    scf = cpcca.squared_covariance_fraction()
    assert isinstance(scf, xr.DataArray)
    assert all(scf <= 1), "Squared covariance fraction is greater than 1"


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

    cpcca = CPCCA(n_modes=2, alpha=alpha, use_pca=use_pca, n_pca_modes="all")
    cpcca.fit(X, Y, "sample")
    tsc_model = cpcca.data["total_squared_covariance"]
    xr.testing.assert_allclose(tsc, tsc_model)


def test_alpha_integer():
    X, Y = generate_well_conditioned_data()

    cpcca = CPCCA(n_modes=2, alpha=1, use_pca=False)
    cpcca.fit(X, Y, "sample")


def test_fit_different_coordinates():
    """Like a lagged CCA scenario"""
    X, Y = generate_well_conditioned_data()
    X = X.isel(sample=slice(0, 99))
    Y = Y.isel(sample=slice(100, 199))
    cpcca = CPCCA(n_modes=2, alpha=1, use_pca=False)
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
    cpcca = CPCCA(n_modes=2, alpha=1, use_pca=False)
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

    cpcca = CPCCA(n_modes=2, alpha=alpha, use_pca=use_pca, n_pca_modes="all")
    cpcca.fit(X, Y, "sample")
    components1, components2 = cpcca.components()
    xr.testing.assert_equal(components1.coords["feature"], X.coords["feature"])
    xr.testing.assert_equal(components2.coords["feature"], Y.coords["feature"])


@pytest.mark.parametrize(
    "correction",
    [(None), ("fdr_bh")],
)
def test_homogeneous_patterns(correction):
    X = generate_random_data((200, 10), seed=123)
    Y = generate_random_data((200, 20), seed=321)

    cpcca = CPCCA(n_modes=10, alpha=1, use_pca=False)
    cpcca.fit(X, Y, "sample")

    _ = cpcca.homogeneous_patterns(correction=correction)


@pytest.mark.parametrize(
    "correction",
    [(None), ("fdr_bh")],
)
def test_heterogeneous_patterns(correction):
    X = generate_random_data((200, 10), seed=123)
    Y = generate_random_data((200, 20), seed=321)

    cpcca = CPCCA(n_modes=10, alpha=1, use_pca=False)
    cpcca.fit(X, Y, "sample")

    _ = cpcca.heterogeneous_patterns(correction=correction)


def test_predict():
    X = generate_random_data((200, 10), seed=123)
    Y = generate_random_data((200, 20), seed=321)

    cpcca = CPCCA(n_modes=10, alpha=0.2, use_pca=False)
    cpcca.fit(X, Y, "sample")

    Xnew = generate_random_data((200, 10), seed=123)

    Ry_pred = cpcca.predict(Xnew)
    _ = cpcca.inverse_transform(Y=Ry_pred)


@pytest.mark.parametrize("engine", ["netcdf4", "zarr"])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_save_load(tmp_path, engine, alpha):
    """Test save/load methods in MCA class, ensuring that we can
    roundtrip the model and get the same results when transforming
    data."""
    X = generate_random_data((200, 10), seed=123)
    Y = generate_random_data((200, 20), seed=321)

    original = CPCCA(alpha=alpha)
    original.fit(X, Y, "sample")

    # Save the CPCCA model
    original.save(tmp_path / "cpcca", engine=engine)

    # Check that the CPCCA model has been saved
    assert (tmp_path / "cpcca").exists()

    # Recreate the model from saved file
    loaded = CPCCA.load(tmp_path / "cpcca", engine=engine)

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
        original.transform(X, Y),
        loaded.transform(X, Y),
    )

    # The loaded model should also be able to inverse_transform new data
    XYr_o = original.inverse_transform(*original.scores())
    XYr_l = loaded.inverse_transform(*loaded.scores())
    assert np.allclose(XYr_o[0], XYr_l[0])
    assert np.allclose(XYr_o[1], XYr_l[1])


def test_serialize_deserialize_dataarray(mock_data_array):
    """Test roundtrip serialization when the model is fit on a DataArray."""
    model = CPCCA()
    model.fit(mock_data_array, mock_data_array, "time")
    dt = model.serialize()
    rebuilt_model = CPCCA.deserialize(dt)
    assert np.allclose(
        model.transform(mock_data_array), rebuilt_model.transform(mock_data_array)
    )


def test_serialize_deserialize_dataset(mock_dataset):
    """Test roundtrip serialization when the model is fit on a Dataset."""
    model = CPCCA()
    model.fit(mock_dataset, mock_dataset, "time")
    dt = model.serialize()
    rebuilt_model = CPCCA.deserialize(dt)
    assert np.allclose(
        model.transform(mock_dataset), rebuilt_model.transform(mock_dataset)
    )
