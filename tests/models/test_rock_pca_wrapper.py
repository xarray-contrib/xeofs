import numpy as np
import pandas as pd
import xarray as xr
import pytest

from xeofs.models import ROCK_PCA
from xeofs.pandas import ROCK_PCA as pdROCK_PCA
from xeofs.xarray import ROCK_PCA as xrROCK_PCA


@pytest.mark.parametrize('n_rot, power, sigma', [
    [2, 0, 1e2],
    [2, 0, 1e3],
    [2, 0, 1e4],
    [2, 0, 1e5],
    [2, 0, 1e6],
    # [2, 1, 1e-2],
    # [2, 1, 1e0],
    # [2, 1, 1e2],
    # [2, 2, 1e-2],
    # [2, 2, 1e0],
    # [2, 2, 1e2],
])
def test_wrapper_solutions(n_rot, power, sigma, sample_array):
    # Solutions of numpy, pandas and xarray wrapper are the same
    X = sample_array
    df = pd.DataFrame(X)
    da = xr.DataArray(X)
    # Perform analysis with all three wrappers
    kwargs = dict(n_rot=n_rot, power=power, sigma=sigma)
    numpy_model = ROCK_PCA(X, **kwargs)
    numpy_model.solve()

    pandas_model = pdROCK_PCA(df, **kwargs)
    pandas_model.solve()

    xarray_model = xrROCK_PCA(da, dim='dim_0', **kwargs)
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
    # EOFs amplitude
    desired_eofs_amplitude = numpy_model.eofs_amplitude()
    actual_pandas_eofs_amplitude = pandas_model.eofs_amplitude().values
    actual_xarray_eofs_amplitude = xarray_model.eofs_amplitude().values
    # EOFs phase
    desired_eofs_phase = numpy_model.eofs_phase()
    actual_pandas_eofs_phase = pandas_model.eofs_phase().values
    actual_xarray_eofs_phase = xarray_model.eofs_phase().values
    # PCs amplitude
    desired_pcs_amplitude = numpy_model.pcs_amplitude()
    actual_pandas_pcs_amplitude = pandas_model.pcs_amplitude().values
    actual_xarray_pcs_amplitude = xarray_model.pcs_amplitude().values
    # PCs phase
    desired_pcs_phase = numpy_model.pcs_phase()
    actual_pandas_pcs_phase = pandas_model.pcs_phase().values
    actual_xarray_pcs_phase = xarray_model.pcs_phase().values

    np.testing.assert_allclose(actual_pandas_expvar, desired_expvar)
    np.testing.assert_allclose(actual_pandas_expvar_ratio, desired_expvar_ratio)
    np.testing.assert_allclose(actual_pandas_pcs, desired_pcs)
    np.testing.assert_allclose(actual_pandas_eofs, desired_eofs)
    np.testing.assert_allclose(actual_pandas_eofs_amplitude, desired_eofs_amplitude)
    np.testing.assert_allclose(actual_pandas_eofs_phase, desired_eofs_phase)
    np.testing.assert_allclose(actual_pandas_pcs_amplitude, desired_pcs_amplitude)
    np.testing.assert_allclose(actual_pandas_pcs_phase, desired_pcs_phase)

    np.testing.assert_allclose(actual_xarray_expvar, desired_expvar)
    np.testing.assert_allclose(actual_xarray_expvar_ratio, desired_expvar_ratio)
    np.testing.assert_allclose(actual_xarray_pcs, desired_pcs)
    np.testing.assert_allclose(actual_xarray_eofs, desired_eofs)
    np.testing.assert_allclose(actual_xarray_eofs_amplitude, desired_eofs_amplitude)
    np.testing.assert_allclose(actual_xarray_eofs_phase, desired_eofs_phase)
    np.testing.assert_allclose(actual_xarray_pcs_amplitude, desired_pcs_amplitude)
    np.testing.assert_allclose(actual_xarray_pcs_phase, desired_pcs_phase)


@pytest.mark.parametrize('n_rot, power, sigma', [
    [3, 0, 1e1],
    [3, 0, 1e2],
    [3, 0, 1e3],
])
def test_wrapper_multivariate_solutions(n_rot, power, sigma, sample_array):
    # Solutions of numpy, pandas and xarray wrapper are the same
    X = sample_array
    X1 = X[:, :10]
    X2 = X[:, 10:]
    df1 = pd.DataFrame(X1)
    df2 = pd.DataFrame(X2)
    da1 = xr.DataArray(X1)
    da2 = xr.DataArray(X2)
    # Perform analysis with all three wrappers
    kwargs = dict(n_rot=n_rot, power=power, sigma=sigma)
    numpy_model = ROCK_PCA([X1, X2], **kwargs)
    numpy_model.solve()

    pandas_model = pdROCK_PCA([df1, df2], **kwargs)
    pandas_model.solve()

    xarray_model = xrROCK_PCA([da1, da2], dim='dim_0', **kwargs)
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
    actual_pandas_pcs = pandas_model.pcs()
    actual_xarray_pcs = xarray_model.pcs()
    # EOFs
    desired_eofs = numpy_model.eofs()
    actual_pandas_eofs = pandas_model.eofs()
    actual_xarray_eofs = xarray_model.eofs()
    # EOFs amplitude
    desired_eofs_amplitude = numpy_model.eofs_amplitude()
    actual_pandas_eofs_amplitude = pandas_model.eofs_amplitude()
    actual_xarray_eofs_amplitude = xarray_model.eofs_amplitude()
    # EOFs phase
    desired_eofs_phase = numpy_model.eofs_phase()
    actual_pandas_eofs_phase = pandas_model.eofs_phase()
    actual_xarray_eofs_phase = xarray_model.eofs_phase()
    # PCs amplitude
    desired_pcs_amplitude = numpy_model.pcs_amplitude()
    actual_pandas_pcs_amplitude = pandas_model.pcs_amplitude()
    actual_xarray_pcs_amplitude = xarray_model.pcs_amplitude()
    # PCs phase
    desired_pcs_phase = numpy_model.pcs_phase()
    actual_pandas_pcs_phase = pandas_model.pcs_phase()
    actual_xarray_pcs_phase = xarray_model.pcs_phase()

    np.testing.assert_allclose(actual_pandas_expvar, desired_expvar)
    np.testing.assert_allclose(actual_pandas_expvar_ratio, desired_expvar_ratio)
    np.testing.assert_allclose(actual_pandas_pcs, desired_pcs)
    np.testing.assert_allclose(actual_pandas_eofs[0], desired_eofs[0])
    np.testing.assert_allclose(actual_pandas_eofs[1], desired_eofs[1])
    np.testing.assert_allclose(actual_pandas_eofs_amplitude[0], desired_eofs_amplitude[0])
    np.testing.assert_allclose(actual_pandas_eofs_amplitude[1], desired_eofs_amplitude[1])

    np.testing.assert_allclose(actual_pandas_eofs_phase[0], desired_eofs_phase[0])
    np.testing.assert_allclose(actual_pandas_eofs_phase[1], desired_eofs_phase[1])
    np.testing.assert_allclose(actual_pandas_pcs_amplitude, desired_pcs_amplitude)
    np.testing.assert_allclose(actual_pandas_pcs_amplitude, desired_pcs_amplitude)
    np.testing.assert_allclose(actual_pandas_pcs_phase, desired_pcs_phase)
    np.testing.assert_allclose(actual_pandas_pcs_phase, desired_pcs_phase)

    np.testing.assert_allclose(actual_xarray_expvar, desired_expvar)
    np.testing.assert_allclose(actual_xarray_expvar_ratio, desired_expvar_ratio)
    np.testing.assert_allclose(actual_xarray_pcs, desired_pcs)
    np.testing.assert_allclose(actual_xarray_eofs[0], desired_eofs[0])
    np.testing.assert_allclose(actual_xarray_eofs[1], desired_eofs[1])
    np.testing.assert_allclose(actual_xarray_eofs_amplitude[0], desired_eofs_amplitude[0])
    np.testing.assert_allclose(actual_xarray_eofs_amplitude[1], desired_eofs_amplitude[1])
    np.testing.assert_allclose(actual_xarray_eofs_phase[0], desired_eofs_phase[0])
    np.testing.assert_allclose(actual_xarray_eofs_phase[1], desired_eofs_phase[1])
    np.testing.assert_allclose(actual_xarray_pcs_amplitude, desired_pcs_amplitude)
    np.testing.assert_allclose(actual_xarray_pcs_amplitude, desired_pcs_amplitude)
    np.testing.assert_allclose(actual_xarray_pcs_phase, desired_pcs_phase)
