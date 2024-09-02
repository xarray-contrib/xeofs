# %%
import timeit

import dask
import eofs
import numpy as np
import xarray as xr
from tqdm import tqdm

import xeofs as xe


# %%
def fit_eofs(X, n_modes=2):
    _ = eofs.xarray.Eof(X)


def fit_xeofs(X, n_modes=2):
    model = xe.single.EOF(n_modes=n_modes, random_state=5)
    model.fit(X, dim="time")


def create_test_data(ns, nf1, nf2, delayed=True, seed=None):
    size = ns * nf1 * nf2
    alpha = size / 1e7
    n_chunks = int(alpha) + 1 if alpha > 1 else 1
    chunk_size = int(np.ceil(ns / n_chunks))

    rng = dask.array.random.default_rng(seed)
    X = rng.standard_normal(
        (ns, nf1, nf2), chunks=(chunk_size, -1, -1), dtype=np.float32
    )
    X = xr.DataArray(
        X,
        dims=("time", "lon", "lat"),
        coords={
            "time": np.arange(ns),
            "lon": np.arange(nf1),
            "lat": np.arange(nf2),
        },
        name="test_data",
    )
    if delayed:
        return X
    else:
        return X.load()


# %%

MAX_MEMORY = {"eofs": 1000, "xeofs": 3000}
n_repeats = 3

n_samples = np.array([10, 20, 60, 100, 200, 600, 1_000, 2_000, 6_000, 10_000])[::-1]
n_features_x = np.array(
    [1, 2, 6, 10, 20, 60, 100, 200, 600, 1_000, 2_000, 6_000, 10_000]
)[::-1]

timings = {
    "data": np.empty((len(n_samples), len(n_features_x))) * np.nan,
    "eofs": np.empty((len(n_samples), len(n_features_x), n_repeats)) * np.nan,
    "xeofs": np.empty((len(n_samples), len(n_features_x), n_repeats)) * np.nan,
    "eofs_dask": np.empty((len(n_samples), len(n_features_x), n_repeats)) * np.nan,
    "xeofs_dask": np.empty((len(n_samples), len(n_features_x), n_repeats)) * np.nan,
}

for i, ns in enumerate(n_samples):
    print(f"n_samples = {ns}")
    for j, nf in tqdm(enumerate(n_features_x), total=len(n_features_x)):
        # Create test data
        X = create_test_data(ns, nf, 10, delayed=True, seed=8)
        nbytes = X.nbytes / 1e6  # in MiB

        # Save data size
        timings["data"][i][j] = X.nbytes

        # Set number of loops
        if 10 < nbytes:
            n_loops = 1
        elif 1 < nbytes and nbytes <= 10:
            n_loops = 10
        elif nbytes <= 1:
            n_loops = 100
        else:
            n_loops = 1

        # eofs / Dask
        try:
            if nbytes < 10_000:  # There seems to be a limit with eofs / Dask
                print("eofs / Dask ...")
                t_eofs = timeit.Timer(lambda: fit_eofs(X))
                timings["eofs_dask"][i][j] = (
                    np.array(t_eofs.repeat(n_repeats, n_loops)) / n_loops
                )
        except Exception as e:
            print(f"Warning: eofs Dask failed for {ns} samples / {nf} features")
            print("Error message: ", e)

        # xeofs / Dask
        try:
            print("xeofs / Dask ...")
            t_xeofs = timeit.Timer(lambda: fit_xeofs(X))
            timings["xeofs_dask"][i][j] = (
                np.array(t_xeofs.repeat(n_repeats, n_loops)) / n_loops
            )
        except Exception as e:
            print(f"Warning: xeofs Dask failed for {ns} samples / {nf} features")
            print("Error message: ", e)

        # eofs / no Dask
        if nbytes < MAX_MEMORY["eofs"]:
            try:
                print("eofs ...")
                X = X.load()
                t_eofs = timeit.Timer(lambda: fit_eofs(X))
                timings["eofs"][i][j] = (
                    np.array(t_eofs.repeat(n_repeats, n_loops)) / n_loops
                )
            except Exception as e:
                print(f"Warning: eofs failed for {ns} samples / {nf} features")
                print("Error message: ", e)

        # xeofs / no Dask
        if nbytes < MAX_MEMORY["xeofs"]:
            try:
                print("xeofs ...")
                X = X.load()
                t_xeofs = timeit.Timer(lambda: fit_xeofs(X))
                timings["xeofs"][i][j] = (
                    np.array(t_xeofs.repeat(n_repeats, n_loops)) / n_loops
                )
            except Exception as e:
                print(f"Warning: xeofs failed for {ns} samples / {nf} features")
                print("Error message: ", e)

# %%
# Save timings
da_size = xr.DataArray(
    timings["data"],
    dims=("n_samples", "n_features"),
    coords={"n_samples": n_samples, "n_features": n_features_x * 10},
    name="nbytes",
)


da1 = xr.DataArray(
    timings["eofs"],
    dims=("n_samples", "n_features", "run"),
    coords={
        "n_samples": n_samples,
        "n_features": n_features_x * 10,
        "run": np.arange(n_repeats),
    },
    name="time",
)
da2 = xr.DataArray(
    timings["xeofs"],
    dims=("n_samples", "n_features", "run"),
    coords={
        "n_samples": n_samples,
        "n_features": n_features_x * 10,
        "run": np.arange(n_repeats),
    },
    name="time",
)

da3 = xr.DataArray(
    timings["eofs_dask"],
    dims=("n_samples", "n_features", "run"),
    coords={
        "n_samples": n_samples,
        "n_features": n_features_x * 10,
        "run": np.arange(n_repeats),
    },
    name="time",
)
da4 = xr.DataArray(
    timings["xeofs_dask"],
    dims=("n_samples", "n_features", "run"),
    coords={
        "n_samples": n_samples,
        "n_features": n_features_x * 10,
        "run": np.arange(n_repeats),
    },
    name="time",
)

# %%
ds = xr.Dataset(
    {"nbytes": da_size, "eofs": da1, "xeofs": da2, "eofs_dask": da3, "xeofs_dask": da4}
)
ds.to_netcdf("timings.nc")


# %%
