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


@pytest.mark.parametrize('n_rot, power, scaling', [
    (2, 1, 0),
    (5, 1, 1),
    (7, 1, 2),
    (2, 2, 0),
    (5, 2, 1),
    (7, 2, 2),
])
def test_wrapper_solutions(n_rot, power, scaling, sample_array):
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
    desired_pcs = numpy_rot.pcs(scaling=scaling)
    actual_pandas_pcs = pandas_rot.pcs(scaling=scaling).values
    actual_xarray_pcs = xarray_rot.pcs(scaling=scaling).values
    # EOFs
    desired_eofs = numpy_rot.eofs(scaling=scaling)
    actual_pandas_eofs = pandas_rot.eofs(scaling=scaling).values
    actual_xarray_eofs = xarray_rot.eofs(scaling=scaling).values
    # EOFs as correlation
    desired_eofs_corr = numpy_rot.eofs_as_correlation()
    actual_pandas_eofs_corr = pandas_rot.eofs_as_correlation()
    actual_xarray_eofs_corr = xarray_rot.eofs_as_correlation()
    # Reconstructed X
    desired_Xrec = numpy_rot.reconstruct_X()
    actual_pandas_Xrec = pandas_rot.reconstruct_X()
    actual_xarray_Xrec = xarray_rot.reconstruct_X()
    # Projection onto EOFs
    desired_proj = numpy_rot.project_onto_eofs(X, scaling=scaling)
    actual_pandas_proj = pandas_rot.project_onto_eofs(df, scaling=scaling)
    actual_xarray_proj = xarray_rot.project_onto_eofs(da, scaling=scaling)

    np.testing.assert_allclose(actual_pandas_expvar, desired_expvar)
    np.testing.assert_allclose(actual_pandas_expvar_ratio, desired_expvar_ratio)
    np.testing.assert_allclose(actual_pandas_pcs, desired_pcs)
    np.testing.assert_allclose(actual_pandas_eofs, desired_eofs)
    np.testing.assert_allclose(actual_pandas_eofs_corr[0], desired_eofs_corr[0])
    np.testing.assert_allclose(actual_pandas_eofs_corr[1], desired_eofs_corr[1])
    np.testing.assert_allclose(actual_pandas_Xrec, desired_Xrec)
    np.testing.assert_allclose(actual_pandas_proj, desired_proj)

    np.testing.assert_allclose(actual_xarray_expvar, desired_expvar)
    np.testing.assert_allclose(actual_xarray_expvar_ratio, desired_expvar_ratio)
    np.testing.assert_allclose(actual_xarray_pcs, desired_pcs)
    np.testing.assert_allclose(actual_xarray_eofs, desired_eofs)
    np.testing.assert_allclose(actual_xarray_eofs_corr[0], desired_eofs_corr[0])
    np.testing.assert_allclose(actual_xarray_eofs_corr[1], desired_eofs_corr[1])
    np.testing.assert_allclose(actual_xarray_Xrec, desired_Xrec)
    np.testing.assert_allclose(actual_xarray_proj, desired_proj)


@pytest.mark.parametrize('n_rot, power, scaling', [
    (2, 1, 0),
    (5, 1, 1),
    (7, 1, 2),
    (2, 2, 0),
    (5, 2, 1),
    (7, 2, 2),
])
def test_wrapper_multivariate_solutions(n_rot, power, scaling, sample_array):
    # Solutions of numpy, pandas and xarray wrapper are the same
    X = sample_array
    X1 = X[:, :10]
    X2 = X[:, 10:]
    df1 = pd.DataFrame(X1)
    df2 = pd.DataFrame(X2)
    da1 = xr.DataArray(X1)
    da2 = xr.DataArray(X2)
    # Perform analysis with all three wrappers
    numpy_model = EOF([X1, X2])
    numpy_model.solve()
    numpy_rot = Rotator(numpy_model, n_rot=n_rot, power=power)

    pandas_model = pdEOF([df1, df2])
    pandas_model.solve()
    pandas_rot = pdRotator(pandas_model, n_rot=n_rot, power=power)

    xarray_model = xrEOF([da1, da2], dim='dim_0')
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
    desired_pcs = numpy_rot.pcs(scaling=scaling)
    actual_pandas_pcs = pandas_rot.pcs(scaling=scaling)
    actual_xarray_pcs = xarray_rot.pcs(scaling=scaling)
    # EOFs
    desired_eofs = numpy_rot.eofs(scaling=scaling)
    actual_pandas_eofs = pandas_rot.eofs(scaling=scaling)
    actual_xarray_eofs = xarray_rot.eofs(scaling=scaling)
    # EOFs as correlation
    desired_eofs_corr = numpy_rot.eofs_as_correlation()
    actual_pandas_eofs_corr = pandas_rot.eofs_as_correlation()
    actual_xarray_eofs_corr = xarray_rot.eofs_as_correlation()
    # Reconstructed X
    desired_Xrec = numpy_rot.reconstruct_X()
    actual_pandas_Xrec = pandas_rot.reconstruct_X()
    actual_xarray_Xrec = xarray_rot.reconstruct_X()
    # Projection onto EOFs
    desired_proj = numpy_rot.project_onto_eofs([X1, X2], scaling=scaling)
    actual_pandas_proj = pandas_rot.project_onto_eofs([df1, df2], scaling=scaling)
    actual_xarray_proj = xarray_rot.project_onto_eofs([da1, da2], scaling=scaling)

    np.testing.assert_allclose(actual_pandas_expvar, desired_expvar)
    np.testing.assert_allclose(actual_pandas_expvar_ratio, desired_expvar_ratio)
    np.testing.assert_allclose(actual_pandas_pcs, desired_pcs)
    np.testing.assert_allclose(actual_pandas_eofs[0], desired_eofs[0])
    np.testing.assert_allclose(actual_pandas_eofs[1], desired_eofs[1])
    np.testing.assert_allclose(actual_pandas_eofs_corr[0][0], desired_eofs_corr[0][0])
    np.testing.assert_allclose(actual_pandas_eofs_corr[0][1], desired_eofs_corr[0][1])
    np.testing.assert_allclose(actual_pandas_eofs_corr[1][0], desired_eofs_corr[1][0])
    np.testing.assert_allclose(actual_pandas_eofs_corr[1][1], desired_eofs_corr[1][1])
    np.testing.assert_allclose(actual_pandas_Xrec[0], desired_Xrec[0])
    np.testing.assert_allclose(actual_pandas_Xrec[1], desired_Xrec[1])
    np.testing.assert_allclose(actual_pandas_proj, desired_proj)

    np.testing.assert_allclose(actual_xarray_expvar, desired_expvar)
    np.testing.assert_allclose(actual_xarray_expvar_ratio, desired_expvar_ratio)
    np.testing.assert_allclose(actual_xarray_pcs, desired_pcs)
    np.testing.assert_allclose(actual_xarray_eofs[0], desired_eofs[0])
    np.testing.assert_allclose(actual_xarray_eofs[1], desired_eofs[1])
    np.testing.assert_allclose(actual_xarray_eofs_corr[0][0], desired_eofs_corr[0][0])
    np.testing.assert_allclose(actual_xarray_eofs_corr[0][1], desired_eofs_corr[0][1])
    np.testing.assert_allclose(actual_xarray_eofs_corr[1][0], desired_eofs_corr[1][0])
    np.testing.assert_allclose(actual_xarray_eofs_corr[1][1], desired_eofs_corr[1][1])
    np.testing.assert_allclose(actual_xarray_Xrec[0], desired_Xrec[0])
    np.testing.assert_allclose(actual_xarray_Xrec[1], desired_Xrec[1])
    np.testing.assert_allclose(actual_xarray_proj, desired_proj)
