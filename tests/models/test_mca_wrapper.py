import numpy as np
import pandas as pd
import xarray as xr
import pytest

from xeofs.models import MCA
from xeofs.xarray import MCA as xrMCA


@pytest.mark.parametrize('scaling, norm', [
    (0, True),
    (1, True),
    (2, True),
    (0, False),
    (1, False),
    (2, False),
])
def test_wrapper_bootstrapper(scaling, norm, sample_array):
    # Bootstrapping runs without errors
    X = sample_array
    df = pd.DataFrame(X)
    da = xr.DataArray(X)
    # Perform analysis with all three wrappers
    numpy_model = MCA(X, X, norm=norm)
    numpy_model.solve()

    xarray_model = xrMCA(da, da, norm=norm, dim='dim_0')
    xarray_model.solve()

    # Numpy
    desired_svals = numpy_model.singular_values()
    desired_expcovar = numpy_model.explained_covariance()
    desired_scf = numpy_model.squared_covariance_fraction()
    desired_svecs = numpy_model.singular_vectors(scaling=scaling)
    desired_pcs = numpy_model.pcs(scaling=scaling)
    desired_hom_pats, desired_pvals_hom = numpy_model.homogeneous_patterns()
    desired_het_pats, desired_pvals_het = numpy_model.heterogeneous_patterns()
    desired_xproj = numpy_model.project_onto_left_singular_vectors(X, scaling=scaling)
    desired_yproj = numpy_model.project_onto_right_singular_vectors(X, scaling=scaling)
    desired_Xrec, desired_Yrec = numpy_model.reconstruct_XY()

    # xarray
    actual_xr_svals = xarray_model.singular_values()
    actual_xr_expcovar = xarray_model.explained_covariance()
    actual_xr_scf = xarray_model.squared_covariance_fraction()
    actual_xr_svecs = xarray_model.singular_vectors(scaling=scaling)
    actual_xr_pcs = xarray_model.pcs(scaling=scaling)
    actual_xr_hom_pats, actual_xr_pvals_hom = xarray_model.homogeneous_patterns()
    actual_xr_het_pats, actual_xr_pvals_het = xarray_model.heterogeneous_patterns()
    actual_xr_xproj = xarray_model.project_onto_left_singular_vectors(da, scaling=scaling)
    actual_xr_yproj = xarray_model.project_onto_right_singular_vectors(da, scaling=scaling)
    actual_xr_Xrec, actual_xr_Yrec = xarray_model.reconstruct_XY()

    # assert consistent xarray results
    np.testing.assert_allclose(desired_svals, actual_xr_svals.values)
    np.testing.assert_allclose(desired_expcovar, actual_xr_expcovar.values)
    np.testing.assert_allclose(desired_scf, actual_xr_scf.values)
    np.testing.assert_allclose(desired_svecs[0], actual_xr_svecs[0].values)
    np.testing.assert_allclose(desired_svecs[1], actual_xr_svecs[1].values)
    np.testing.assert_allclose(desired_pcs[0], actual_xr_pcs[0].values)
    np.testing.assert_allclose(desired_pcs[1], actual_xr_pcs[1].values)
    np.testing.assert_allclose(desired_hom_pats[0], actual_xr_hom_pats[0].values)
    np.testing.assert_allclose(desired_hom_pats[1], actual_xr_hom_pats[1].values)
    np.testing.assert_allclose(desired_het_pats[0], actual_xr_het_pats[0].values)
    np.testing.assert_allclose(desired_het_pats[1], actual_xr_het_pats[1].values)
    np.testing.assert_allclose(desired_xproj, actual_xr_xproj.values)
    np.testing.assert_allclose(desired_yproj, actual_xr_yproj.values)
