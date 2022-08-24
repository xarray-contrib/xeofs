import numpy as np
import pandas as pd
import xarray as xr
import pytest

from xeofs.models.mca import MCA
from xeofs.pandas.mca import MCA as pdMCA
from xeofs.xarray.mca import MCA as xrMCA
from xeofs.models.mca_rotator import MCA_Rotator
from xeofs.pandas.mca_rotator import MCA_Rotator as pdMCA_Rotator
from xeofs.xarray.mca_rotator import MCA_Rotator as xrMCA_Rotator


@pytest.mark.parametrize('n_rot, power, scaling', [
    (3, 1, 0),
    (5, 1, 1),
    (7, 1, 2),
    (3, 2, 0),
    (5, 2, 1),
    (7, 2, 2),
])
def test_wrapper_solutions(n_rot, power, scaling, sample_array):
    # Solutions of numpy, pandas and xarray wrapper are the same
    X = sample_array
    print(X.shape)
    X1 = X[:, :9]
    X2 = X[:, 9:]
    df1 = pd.DataFrame(X1)
    df2 = pd.DataFrame(X2)
    da1 = xr.DataArray(X1)
    da2 = xr.DataArray(X2)
    # Perform analysis with all three wrappers
    numpy_model = MCA(X1, X2)
    numpy_model.solve()
    numpy_rot = MCA_Rotator(n_rot=n_rot, power=power)
    numpy_rot.rotate(numpy_model)

    pandas_model = pdMCA(df1, df2)
    pandas_model.solve()
    pandas_rot = pdMCA_Rotator(n_rot=n_rot, power=power)
    pandas_rot.rotate(pandas_model)

    xarray_model = xrMCA(da1, da2, dim='dim_0')
    xarray_model.solve()
    xarray_rot = xrMCA_Rotator(n_rot=n_rot, power=power)
    xarray_rot.rotate(xarray_model)

    # Explained variance
    desired_expvar = numpy_rot.explained_covariance()
    actual_pandas_expvar = pandas_rot.explained_covariance().values
    actual_xarray_expvar = xarray_rot.explained_covariance().values
    # Explained variance ratio
    desired_expvar_ratio = numpy_rot.squared_covariance_fraction()
    actual_pandas_expvar_ratio = pandas_rot.squared_covariance_fraction().values
    actual_xarray_expvar_ratio = xarray_rot.squared_covariance_fraction().values
    # PCs
    desired_pcs = numpy_rot.pcs(scaling=scaling)
    actual_pandas_pcs = pandas_rot.pcs(scaling=scaling)
    actual_xarray_pcs = xarray_rot.pcs(scaling=scaling)
    # EOFs
    desired_singular_vectors = numpy_rot.singular_vectors(scaling=scaling)
    actual_pandas_singular_vectors = pandas_rot.singular_vectors(scaling=scaling)
    actual_xarray_singular_vectors = xarray_rot.singular_vectors(scaling=scaling)
    # homogeneous_patterns
    desired_hom_pat, _ = numpy_rot.homogeneous_patterns()
    actual_pandas_hom_pat, _ = pandas_rot.homogeneous_patterns()
    actual_xarray_hom_pat, _ = xarray_rot.homogeneous_patterns()
    # heterogeneous_patterns
    desired_het_pat, _ = numpy_rot.heterogeneous_patterns()
    actual_pandas_het_pat, _ = pandas_rot.heterogeneous_patterns()
    actual_xarray_het_pat, _ = xarray_rot.heterogeneous_patterns()
    # Reconstructed X
    desired_Xrec, desired_Yrec = numpy_rot.reconstruct_XY()
    actual_pandas_Xrec, actual_pandas_Yrec = pandas_rot.reconstruct_XY()
    actual_xarray_Xrec, actual_xarray_Yrec = xarray_rot.reconstruct_XY()
    # Projection left singular_vectors
    desired_projx = numpy_rot.project_onto_left_singular_vectors(X1, scaling=scaling)
    actual_pandas_projx = pandas_rot.project_onto_left_singular_vectors(df1, scaling=scaling)
    actual_xarray_projx = xarray_rot.project_onto_left_singular_vectors(da1, scaling=scaling)
    # Projection right singular_vectors
    desired_projy = numpy_rot.project_onto_right_singular_vectors(X2, scaling=scaling)
    actual_pandas_projy = pandas_rot.project_onto_right_singular_vectors(df2, scaling=scaling)
    actual_xarray_projy = xarray_rot.project_onto_right_singular_vectors(da2, scaling=scaling)

    np.testing.assert_allclose(actual_pandas_expvar, desired_expvar)
    np.testing.assert_allclose(actual_pandas_expvar_ratio, desired_expvar_ratio)
    np.testing.assert_allclose(actual_pandas_pcs, desired_pcs)
    np.testing.assert_allclose(actual_pandas_singular_vectors[0], desired_singular_vectors[0])
    np.testing.assert_allclose(actual_pandas_singular_vectors[1], desired_singular_vectors[1])
    np.testing.assert_allclose(actual_pandas_hom_pat[0], desired_hom_pat[0])
    np.testing.assert_allclose(actual_pandas_hom_pat[1], desired_hom_pat[1])
    np.testing.assert_allclose(actual_pandas_het_pat[0], desired_het_pat[0])
    np.testing.assert_allclose(actual_pandas_het_pat[1], desired_het_pat[1])
    np.testing.assert_allclose(actual_pandas_Xrec, desired_Xrec)
    np.testing.assert_allclose(actual_pandas_Yrec, desired_Yrec)
    np.testing.assert_allclose(actual_pandas_projx, desired_projx)
    np.testing.assert_allclose(actual_pandas_projy, desired_projy)

    np.testing.assert_allclose(actual_xarray_expvar, desired_expvar)
    np.testing.assert_allclose(actual_xarray_expvar_ratio, desired_expvar_ratio)
    np.testing.assert_allclose(actual_xarray_pcs, desired_pcs)
    np.testing.assert_allclose(actual_xarray_singular_vectors[0], desired_singular_vectors[0])
    np.testing.assert_allclose(actual_xarray_singular_vectors[1], desired_singular_vectors[1])
    np.testing.assert_allclose(actual_xarray_hom_pat[0], desired_hom_pat[0])
    np.testing.assert_allclose(actual_xarray_hom_pat[1], desired_hom_pat[1])
    np.testing.assert_allclose(actual_xarray_het_pat[0], desired_het_pat[0])
    np.testing.assert_allclose(actual_xarray_het_pat[1], desired_het_pat[1])
    np.testing.assert_allclose(actual_xarray_Xrec, desired_Xrec)
    np.testing.assert_allclose(actual_xarray_Yrec, desired_Yrec)
    np.testing.assert_allclose(actual_xarray_projx, desired_projx)
    np.testing.assert_allclose(actual_xarray_projy, desired_projy)


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
    X1 = X[:, :9]
    X2 = X[:, 9:]
    df1 = pd.DataFrame(X1)
    df2 = pd.DataFrame(X2)
    da1 = xr.DataArray(X1)
    da2 = xr.DataArray(X2)
    # Perform analysis with all three wrappers
    numpy_model = MCA([X1, X2], X2)
    numpy_model.solve()
    numpy_rot = MCA_Rotator(n_rot=n_rot, power=power)
    numpy_rot.rotate(numpy_model)

    pandas_model = pdMCA([df1, df2], df2)
    pandas_model.solve()
    pandas_rot = pdMCA_Rotator(n_rot=n_rot, power=power)
    pandas_rot.rotate(pandas_model)

    xarray_model = xrMCA([da1, da2], da2, dim='dim_0')
    xarray_model.solve()
    xarray_rot = xrMCA_Rotator(n_rot=n_rot, power=power)
    xarray_rot.rotate(xarray_model)

    # Explained variance
    desired_expvar = numpy_rot.explained_covariance()
    actual_pandas_expvar = pandas_rot.explained_covariance().values
    actual_xarray_expvar = xarray_rot.explained_covariance().values
    # Explained variance ratio
    desired_expvar_ratio = numpy_rot.squared_covariance_fraction()
    actual_pandas_expvar_ratio = pandas_rot.squared_covariance_fraction().values
    actual_xarray_expvar_ratio = xarray_rot.squared_covariance_fraction().values
    # PCs
    desired_pcs = numpy_rot.pcs(scaling=scaling)
    actual_pandas_pcs = pandas_rot.pcs(scaling=scaling)
    actual_xarray_pcs = xarray_rot.pcs(scaling=scaling)
    # EOFs
    desired_singular_vectors = numpy_rot.singular_vectors(scaling=scaling)
    actual_pandas_singular_vectors = pandas_rot.singular_vectors(scaling=scaling)
    actual_xarray_singular_vectors = xarray_rot.singular_vectors(scaling=scaling)
    # homogeneous_patterns
    desired_hom_pat, _ = numpy_rot.homogeneous_patterns()
    actual_pandas_hom_pat, _ = pandas_rot.homogeneous_patterns()
    actual_xarray_hom_pat, _ = xarray_rot.homogeneous_patterns()
    # heterogeneous_patterns
    desired_het_pat, _ = numpy_rot.heterogeneous_patterns()
    actual_pandas_het_pat, _ = pandas_rot.heterogeneous_patterns()
    actual_xarray_het_pat, _ = xarray_rot.heterogeneous_patterns()
    # Reconstructed X
    desired_Xrec, desired_Yrec = numpy_rot.reconstruct_XY()
    actual_pandas_Xrec, actual_pandas_Yrec = pandas_rot.reconstruct_XY()
    actual_xarray_Xrec, actual_xarray_Yrec = xarray_rot.reconstruct_XY()
    # Projection left singular_vectors
    desired_projx = numpy_rot.project_onto_left_singular_vectors([X1, X2], scaling=scaling)
    actual_pandas_projx = pandas_rot.project_onto_left_singular_vectors([df1, df2], scaling=scaling)
    actual_xarray_projx = xarray_rot.project_onto_left_singular_vectors([da1, da2], scaling=scaling)
    # Projection right singular_vectors
    desired_projy = numpy_rot.project_onto_right_singular_vectors(X2, scaling=scaling)
    actual_pandas_projy = pandas_rot.project_onto_right_singular_vectors(df2, scaling=scaling)
    actual_xarray_projy = xarray_rot.project_onto_right_singular_vectors(da2, scaling=scaling)

    np.testing.assert_allclose(actual_pandas_expvar, desired_expvar)
    np.testing.assert_allclose(actual_pandas_expvar_ratio, desired_expvar_ratio)
    np.testing.assert_allclose(actual_pandas_pcs, desired_pcs)
    np.testing.assert_allclose(actual_pandas_singular_vectors[0][0], desired_singular_vectors[0][0])
    np.testing.assert_allclose(actual_pandas_singular_vectors[0][1], desired_singular_vectors[0][1])
    np.testing.assert_allclose(actual_pandas_singular_vectors[1], desired_singular_vectors[1])
    np.testing.assert_allclose(actual_pandas_hom_pat[0][0], desired_hom_pat[0][0])
    np.testing.assert_allclose(actual_pandas_hom_pat[0][1], desired_hom_pat[0][1])
    np.testing.assert_allclose(actual_pandas_hom_pat[1], desired_hom_pat[1])
    np.testing.assert_allclose(actual_pandas_het_pat[0][0], desired_het_pat[0][0])
    np.testing.assert_allclose(actual_pandas_het_pat[0][1], desired_het_pat[0][1])
    np.testing.assert_allclose(actual_pandas_het_pat[1], desired_het_pat[1])
    np.testing.assert_allclose(actual_pandas_Xrec[0], desired_Xrec[0])
    np.testing.assert_allclose(actual_pandas_Xrec[1], desired_Xrec[1])
    np.testing.assert_allclose(actual_pandas_Yrec, desired_Yrec)
    np.testing.assert_allclose(actual_pandas_projx, desired_projx)
    np.testing.assert_allclose(actual_pandas_projy, desired_projy)

    np.testing.assert_allclose(actual_xarray_expvar, desired_expvar)
    np.testing.assert_allclose(actual_xarray_expvar_ratio, desired_expvar_ratio)
    np.testing.assert_allclose(actual_xarray_pcs, desired_pcs)
    np.testing.assert_allclose(actual_xarray_singular_vectors[0][0], desired_singular_vectors[0][0])
    np.testing.assert_allclose(actual_xarray_singular_vectors[0][1], desired_singular_vectors[0][1])
    np.testing.assert_allclose(actual_xarray_singular_vectors[1], desired_singular_vectors[1])
    np.testing.assert_allclose(actual_xarray_hom_pat[0][0], desired_hom_pat[0][0])
    np.testing.assert_allclose(actual_xarray_hom_pat[0][1], desired_hom_pat[0][1])
    np.testing.assert_allclose(actual_xarray_hom_pat[1], desired_hom_pat[1])
    np.testing.assert_allclose(actual_xarray_het_pat[0][0], desired_het_pat[0][0])
    np.testing.assert_allclose(actual_xarray_het_pat[0][1], desired_het_pat[0][1])
    np.testing.assert_allclose(actual_xarray_het_pat[1], desired_het_pat[1])
    np.testing.assert_allclose(actual_xarray_Xrec[0], desired_Xrec[0])
    np.testing.assert_allclose(actual_xarray_Xrec[1], desired_Xrec[1])
    np.testing.assert_allclose(actual_xarray_Yrec, desired_Yrec)
    np.testing.assert_allclose(actual_xarray_projx, desired_projx)
    np.testing.assert_allclose(actual_xarray_projy, desired_projy)
