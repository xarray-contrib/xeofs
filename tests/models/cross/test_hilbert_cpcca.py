import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xeofs.cross import HilbertCPCCA

from ...utilities import skip_if_missing_engine


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


@pytest.mark.parametrize("use_pca", [True, False])
def test_singular_values(use_pca):
    """Test that the singular values of the Hilbert CCA are less than 1."""
    X, Y = generate_well_conditioned_data()
    cpcca = HilbertCPCCA(n_modes=2, alpha=0.0, use_pca=use_pca, n_pca_modes=2)
    cpcca.fit(X, Y, "sample")
    s_values = cpcca.data["singular_values"]

    # Singular values are the canonical correlations, so they should be less than 1
    assert np.all(s_values <= 1)


# Currently, netCDF4 does not support complex numbers, so skip this test
@pytest.mark.parametrize("engine", ["h5netcdf", "zarr"])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_save_load_with_data(tmp_path, engine, alpha):
    """Test save/load methods in CPCCA class, ensuring that we can
    roundtrip the model and get the same results."""
    skip_if_missing_engine(engine)

    X = generate_random_data((200, 10), seed=123)
    Y = generate_random_data((200, 20), seed=321)

    original = HilbertCPCCA(alpha=alpha)
    original.fit(X, Y, "sample")

    # Save the CPCCA model
    original.save(tmp_path / "cpcca", engine=engine, save_data=True)

    # Check that the CPCCA model has been saved
    assert (tmp_path / "cpcca").exists()

    # Recreate the model from saved file
    loaded = HilbertCPCCA.load(tmp_path / "cpcca", engine=engine)

    # Check that the params and DataContainer objects match
    assert original.get_params() == loaded.get_params()
    assert all([key in loaded.data for key in original.data])
    for key in original.data:
        assert loaded.data[key].equals(original.data[key])

    # Test that the recreated model can compute the SCF
    assert np.allclose(
        original.squared_covariance_fraction(), loaded.squared_covariance_fraction()
    )

    # Test that the recreated model can compute the components amplitude
    A1_original, A2_original = original.components_amplitude()
    A1_loaded, A2_loaded = loaded.components_amplitude()
    assert np.allclose(A1_original, A1_loaded)
    assert np.allclose(A2_original, A2_loaded)

    # Test that the recreated model can compute the components phase
    P1_original, P2_original = original.components_phase()
    P1_loaded, P2_loaded = loaded.components_phase()
    assert np.allclose(P1_original, P1_loaded)
    assert np.allclose(P2_original, P2_loaded)
