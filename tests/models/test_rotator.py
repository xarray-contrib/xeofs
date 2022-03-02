import numpy as np
import pytest
from pytest import approx
from numpy.testing import assert_allclose, assert_raises

from xeofs.models.eof import EOF
from xeofs.models.rotator import Rotator


@pytest.mark.parametrize('n_modes', [0, 1])
def test_invalid_rotation(n_modes, sample_array):
    # Choosing less than 2 modes
    X = sample_array
    model = EOF(X)
    model.solve()
    with pytest.raises(Exception):
        _ = Rotator(model=model, n_rot=n_modes)


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


def test_eofs_as_correlation(sample_array):
    # Correlation coefficients are between -1 and 1
    # p values are between 0 and 1
    data_no_nan = sample_array[:, ~np.isnan(sample_array).all(axis=0)]
    model = EOF(data_no_nan)
    model.solve()
    rot = Rotator(model=model, n_rot=5)
    corr, pvals = rot.eofs_as_correlation()
    assert (abs(corr) <= 1).all()
    assert (pvals >= 0).all()
    assert (pvals <= 1).all()


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


@pytest.mark.parametrize('n_rot', [2, 5])
def test_reconstruct_X(n_rot, sample_array):
    # Reconstructed data is the same before and after rotation.
    model = EOF(sample_array, n_modes=n_rot)
    model.solve()
    rot = Rotator(model, n_rot=n_rot)

    Xrec = model.reconstruct_X()
    Xrec_rot = rot.reconstruct_X()
    # X values range between -40 and 40, absolute tolerance of 0.5 seems OK.
    np.testing.assert_allclose(Xrec_rot, Xrec, atol=.5)


@pytest.mark.parametrize('norm, scaling', [
    (False, 0),
    (False, 1),
    (False, 2),
    (True, 0),
    (True, 1),
    (True, 2),
])
def test_project_onto_eofs(norm, scaling, sample_array):
    # Projection of original data and PCs are the same.
    model = EOF(sample_array, norm=norm)
    model.solve()
    rot = Rotator(model, n_rot=2)
    pcs = rot.pcs(scaling=scaling)
    projections = rot.project_onto_eofs(sample_array, scaling=scaling)
    np.testing.assert_allclose(projections, pcs)
