import numpy as np
import pandas as pd
import xarray as xr
import pytest

from xeofs.models import EOF
from xeofs.xarray import EOF as xrEOF
from xeofs.models import Bootstrapper
from xeofs.xarray import Bootstrapper as xrBootstrapper


def test_wrapper_bootstrapper(sample_array):
    # Bootstrapping runs without errors
    X = sample_array
    df = pd.DataFrame(X)
    da = xr.DataArray(X)
    # Perform analysis with all three wrappers
    numpy_model = EOF(X)
    numpy_model.solve()

    xarray_model = xrEOF(da, dim='dim_0')
    xarray_model.solve()

    # Numpy Bootstrapper
    bs_np = Bootstrapper(n_boot=20, alpha=.05)
    bs_np.bootstrap(numpy_model)
    params = bs_np.get_params()
    n_sig = bs_np.n_significant_modes()
    expvar, expvar_mask = bs_np.explained_variance()
    eofs, eofs_mask = bs_np.eofs()
    pcs, pcs_mask = bs_np.pcs()

    # xarray Bootstrapper
    bs_xr = xrBootstrapper(n_boot=20, alpha=.05)
    bs_xr.bootstrap(xarray_model)
    params = bs_xr.get_params()
    n_sig = bs_xr.n_significant_modes()
    expvar, expvar_mask = bs_xr.explained_variance()
    eofs, eofs_mask = bs_xr.eofs()
    pcs, pcs_mask = bs_xr.pcs()
