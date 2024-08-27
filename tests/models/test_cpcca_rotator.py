import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xeofs.models import ContinuumPowerCCA, CPCCARotator


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


@pytest.mark.parametrize(
    "correction",
    [(None), ("fdr_bh")],
)
def test_homogeneous_patterns(correction):
    X = generate_random_data((200, 10), seed=123)
    Y = generate_random_data((200, 20), seed=321)

    cpcca = ContinuumPowerCCA(n_modes=10, alpha=1, use_pca=False)
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

    cpcca = ContinuumPowerCCA(n_modes=10, alpha=1, use_pca=False)
    cpcca.fit(X, Y, "sample")

    rotator = CPCCARotator(n_modes=4)
    rotator.fit(cpcca)

    _ = rotator.heterogeneous_patterns(correction=correction)
