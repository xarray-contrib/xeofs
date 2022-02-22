import numpy as np
import pandas as pd
import xarray as xr
import pytest

from xeofs.models.eof import EOF
from xeofs.pandas.eof import EOF as pdEOF
from xeofs.xarray.eof import EOF as xrEOF
from xeofs.models.rotator import Rotator
from xeofs.pandas.rotator import Rotator as pdRotator
from xeofs.xarray.rotator import Rotator as xrRotator


@pytest.mark.parametrize('n_rot, power', [
    (2, 1),
    (5, 1),
    (7, 1),
    (2, 2),
    (5, 2),
    (7, 2),
])
def test_wrapper_solutions(n_rot, power, sample_array):
    # Solutions of numpy, pandas and xarray wrapper are the same
    X = sample_array
    df = pd.DataFrame(X)
    da = xr.DataArray(X)
    # Perform analysis with all three wrappers
    numpy_model = EOF(X)
    numpy_model.solve()
    numpy_rot = Rotator(numpy_model, n_rot=n_rot, power=power)

    pandas_model = pdEOF(df)
    pandas_model.solve()
    pandas_rot = pdRotator(pandas_model, n_rot=n_rot, power=power)

    xarray_model = xrEOF(da, dim='dim_0')
    xarray_model.solve()
    xarray_rot = xrRotator(xarray_model, n_rot=n_rot, power=power)

    # Explained variance
    desired_expvar = numpy_rot.explained_variance()
    actual_pandas_expvar = pandas_rot.explained_variance().squeeze()
    actual_xarray_expvar = xarray_rot.explained_variance()
    # Explained variance ratio
    desired_expvar_ratio = numpy_rot.explained_variance_ratio()
    actual_pandas_expvar_ratio = pandas_rot.explained_variance_ratio().squeeze()
    actual_xarray_expvar_ratio = xarray_rot.explained_variance_ratio()
    # PCs
    desired_pcs = numpy_rot.pcs()
    actual_pandas_pcs = pandas_rot.pcs().values
    actual_xarray_pcs = xarray_rot.pcs().values
    # EOFs
    desired_eofs = numpy_rot.eofs()
    actual_pandas_eofs = pandas_rot.eofs().values
    actual_xarray_eofs = xarray_rot.eofs().values
    # EOFs as correlation
    desired_eofs_corr = numpy_rot.eofs_as_correlation()
    actual_pandas_eofs_corr = pandas_rot.eofs_as_correlation()
    actual_xarray_eofs_corr = xarray_rot.eofs_as_correlation()

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
