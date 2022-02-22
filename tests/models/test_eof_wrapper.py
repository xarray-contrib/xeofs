import numpy as np
import pandas as pd
import xarray as xr
import pytest

from xeofs.models.eof import EOF
from xeofs.pandas.eof import EOF as pdEOF
from xeofs.xarray.eof import EOF as xrEOF


def test_wrapper_solutions(sample_array):
    # Solutions of numpy, pandas and xarray wrapper are the same
    X = sample_array
    df = pd.DataFrame(X)
    da = xr.DataArray(X)
    # Perform analysis with all three wrappers
    numpy_model = EOF(X)
    numpy_model.solve()

    pandas_model = pdEOF(df)
    pandas_model.solve()

    xarray_model = xrEOF(da, dim='dim_0')
    xarray_model.solve()

    # Explained variance
    desired_expvar = numpy_model.explained_variance()
    actual_pandas_expvar = pandas_model.explained_variance().squeeze()
    actual_xarray_expvar = xarray_model.explained_variance()
    # Explained variance ratio
    desired_expvar_ratio = numpy_model.explained_variance_ratio()
    actual_pandas_expvar_ratio = pandas_model.explained_variance_ratio().squeeze()
    actual_xarray_expvar_ratio = xarray_model.explained_variance_ratio()
    # PCs
    desired_pcs = numpy_model.pcs()
    actual_pandas_pcs = pandas_model.pcs().values
    actual_xarray_pcs = xarray_model.pcs().values
    # EOFs
    desired_eofs = numpy_model.eofs()
    actual_pandas_eofs = pandas_model.eofs().values
    actual_xarray_eofs = xarray_model.eofs().values
    # EOFs as correlation
    desired_eofs_corr = numpy_model.eofs_as_correlation()
    actual_pandas_eofs_corr = pandas_model.eofs_as_correlation()
    actual_xarray_eofs_corr = xarray_model.eofs_as_correlation()

    np.testing.assert_allclose(actual_pandas_expvar, desired_expvar)
    np.testing.assert_allclose(actual_pandas_expvar_ratio, desired_expvar_ratio)
    np.testing.assert_allclose(actual_pandas_pcs, desired_pcs)
    np.testing.assert_allclose(actual_pandas_eofs, desired_eofs)
    np.testing.assert_allclose(actual_pandas_eofs_corr[0], desired_eofs_corr[0])
    np.testing.assert_allclose(actual_pandas_eofs_corr[1], desired_eofs_corr[1])

    np.testing.assert_allclose(actual_xarray_expvar, desired_expvar)
    np.testing.assert_allclose(actual_xarray_expvar_ratio, desired_expvar_ratio)
    np.testing.assert_allclose(actual_xarray_pcs, desired_pcs)
    np.testing.assert_allclose(actual_xarray_eofs, desired_eofs)
    np.testing.assert_allclose(actual_xarray_eofs_corr[0], desired_eofs_corr[0])
    np.testing.assert_allclose(actual_xarray_eofs_corr[1], desired_eofs_corr[1])
