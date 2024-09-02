import dask.array.random as da
import numpy as np
import pytest
import xarray as xr

from xeofs.cross import CPCCA, CPCCARotator


def generate_random_data(shape, lazy=False, seed=142):
    rng = np.random.default_rng(seed)
    if lazy:
        return xr.DataArray(
            da.random(shape, chunks=(5, 5)),
            dims=["sample", "feature"],
            coords={"sample": np.arange(shape[0]), "feature": np.arange(shape[1])},
        )
    else:
        return xr.DataArray(
            rng.random(shape),
            dims=["sample", "feature"],
            coords={"sample": np.arange(shape[0]), "feature": np.arange(shape[1])},
        )


@pytest.mark.parametrize(
    "correction",
    [(None), ("fdr_bh")],
)
def test_homogeneous_patterns(correction):
    X = generate_random_data((200, 10), seed=123)
    Y = generate_random_data((200, 20), seed=321)

    cpcca = CPCCA(n_modes=10, alpha=1, use_pca=False)
    cpcca.fit(X, Y, "sample")

    rotator = CPCCARotator(n_modes=4)
    rotator.fit(cpcca)

    _ = rotator.homogeneous_patterns(correction=correction)


@pytest.mark.parametrize(
    "correction",
    [(None), ("fdr_bh")],
)
def test_heterogeneous_patterns(correction):
    X = generate_random_data((200, 10), seed=123)
    Y = generate_random_data((200, 20), seed=321)

    cpcca = CPCCA(n_modes=10, alpha=1, use_pca=False)
    cpcca.fit(X, Y, "sample")

    rotator = CPCCARotator(n_modes=4)
    rotator.fit(cpcca)

    _ = rotator.heterogeneous_patterns(correction=correction)


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
    X = generate_random_data((200, 10), seed=123)
    Y = generate_random_data((200, 20), seed=321)

    cpcca = CPCCA(n_modes=10, alpha=alpha, use_pca=use_pca, n_pca_modes="all")
    cpcca.fit(X, Y, "sample")
    rotator = CPCCARotator(n_modes=10)
    rotator.fit(cpcca)

    scf = rotator.squared_covariance_fraction()
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
def test_squared_covariance_fraction_conserved(alpha, use_pca):
    X = generate_random_data((200, 10), seed=123)
    Y = generate_random_data((200, 20), seed=321)

    cpcca = CPCCA(n_modes=10, alpha=alpha, use_pca=use_pca, n_pca_modes="all")
    cpcca.fit(X, Y, "sample")

    n_rot_modes = 5
    rotator = CPCCARotator(n_modes=n_rot_modes, power=1)
    rotator.fit(cpcca)

    scf = rotator.squared_covariance_fraction()
    scf_rot = rotator.squared_covariance_fraction()

    scf_sum = scf.sel(mode=slice(1, n_rot_modes)).sum()
    scf_rot_sum = scf_rot.sel(mode=slice(1, n_rot_modes)).sum()

    xr.testing.assert_allclose(scf_sum, scf_rot_sum)
