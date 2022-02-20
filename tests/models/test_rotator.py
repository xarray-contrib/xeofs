import numpy as np
import pandas as pd
import xarray as xr
import pytest
from pytest import approx
from numpy.testing import assert_allclose, assert_raises

from xeofs.models.eof import EOF
from xeofs.pandas.eof import EOF as pdEOF
from xeofs.xarray.eof import EOF as xrEOF
from xeofs.models.rotator import Rotator
from xeofs.pandas.rotator import Rotator as pdRotator
from xeofs.xarray.rotator import Rotator as xrRotator


@pytest.mark.parametrize('n_modes', [0, 1])
def test_invalid_rotation(n_modes, sample_array):
    # Choosing less than 2 modes
    X = sample_array
    model = EOF(X)
    model.solve()
    with pytest.raises(Exception):
        rot = Rotator(model=model, n_rot=n_modes)

    
@pytest.mark.parametrize('n_modes', [2, 5, 7])
def test_explained_variance(n_modes, sample_array):
    # Amount of explained variance is same before and after Varimax rotation
    X = sample_array
    model = EOF(X)
    model.solve()
    rot = Rotator(model=model, n_rot=n_modes)
    expvar = model.explained_variance()[:n_modes]
    rot_expvar = rot.explained_variance()

    assert approx(expvar.sum(), 0.01) == approx(rot_expvar.sum(), 0.01)


@pytest.mark.parametrize('n_modes', [2, 5, 7])
def test_explained_variance_ratio(n_modes, sample_array):
    # Amount of explained variance is same before and after Varimax rotation
    X = sample_array
    model = EOF(X)
    model.solve()
    rot = Rotator(model=model, n_rot=n_modes)
    expvar_ratio = model.explained_variance_ratio()[:n_modes]
    rot_expvar_ratio = rot.explained_variance_ratio()

    assert approx(expvar_ratio.sum(), 0.01) == approx(rot_expvar_ratio.sum(), 0.01)


@pytest.mark.parametrize('n_modes', [2, 5, 7])
def test_pcs_uncorrelated(n_modes, sample_array):
    # PCs are uncorrelated after orthogonal Varimax rotation
    X = sample_array
    model = EOF(X)
    model.solve()
    pcs = model.pcs()[:, :n_modes]
    rot = Rotator(model=model, n_rot=n_modes)
    rpcs = rot.pcs()

    assert_allclose(pcs.T @ pcs, np.identity(n_modes), atol=1e-3)
    assert_allclose(rpcs.T @ rpcs, np.identity(n_modes), atol=1e-3)


@pytest.mark.parametrize('n_modes, power', [
    (2, 1),
    (5, 1),
    (7, 1),
    (2, 2),
    (5, 2),
    (7, 2),
])
def test_relaxed_orthogonal_contraint(n_modes, power, sample_array):
    # EOFs are not orthogonal after rotation
    X = sample_array
    model = EOF(X)
    model.solve()
    rot_var = Rotator(model=model, n_rot=n_modes, power=power)
    rot_pro = Rotator(model=model, n_rot=n_modes, power=power)
    eofs_var = rot_var._eofs
    eofs_pro = rot_pro._eofs

    actual_var = eofs_var.T @ eofs_var
    actual_pro = eofs_pro.T @ eofs_pro
    desired = np.identity(n_modes)

    assert_raises(AssertionError, assert_allclose, actual_var, desired, atol=1e-3)
    assert_raises(AssertionError, assert_allclose, actual_pro, desired, atol=1e-3)


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

    numpy_model = EOF(X)
    numpy_model.solve()
    numpy_rot = Rotator(numpy_model, n_rot=n_rot, power=power)

    pandas_model = pdEOF(df)
    pandas_model.solve()
    pandas_rot = pdRotator(pandas_model, n_rot=n_rot, power=power)

    xarray_model = xrEOF(da, dim='dim_0')
    xarray_model.solve()
    xarray_rot = xrRotator(xarray_model, n_rot=n_rot, power=power)

    desired_expvar = numpy_rot.explained_variance()
    actual_pandas_expvar = pandas_rot.explained_variance().squeeze()
    actual_xarray_expvar = xarray_rot.explained_variance()

    desired_expvar_ratio = numpy_rot.explained_variance_ratio()
    actual_pandas_expvar_ratio = pandas_rot.explained_variance_ratio().squeeze()
    actual_xarray_expvar_ratio = xarray_rot.explained_variance_ratio()

    desired_pcs = numpy_rot.pcs()
    actual_pandas_pcs = pandas_rot.pcs().values
    actual_xarray_pcs = xarray_rot.pcs().values

    desired_eofs = numpy_rot.eofs()
    actual_pandas_eofs = pandas_rot.eofs().values
    actual_xarray_eofs = xarray_rot.eofs().values

    np.testing.assert_allclose(actual_pandas_expvar, desired_expvar)
    np.testing.assert_allclose(actual_pandas_expvar_ratio, desired_expvar_ratio)
    np.testing.assert_allclose(actual_pandas_pcs, desired_pcs)
    np.testing.assert_allclose(actual_pandas_eofs, desired_eofs)

    np.testing.assert_allclose(actual_xarray_expvar, desired_expvar)
    np.testing.assert_allclose(actual_xarray_expvar_ratio, desired_expvar_ratio)
    np.testing.assert_allclose(actual_xarray_pcs, desired_pcs)
    np.testing.assert_allclose(actual_xarray_eofs, desired_eofs)
