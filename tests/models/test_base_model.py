import numpy as np
import xarray as xr
import pytest
import dask.array as da
import warnings
from numpy.testing import assert_allclose

from xeofs.models._base_model import EOF, ComplexEOF

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


@pytest.mark.parametrize('method, standardize, use_weights', [
    ('EOF', False, None),
    ('EOF', True, None)
])
def test_solution(method, standardize, use_weights, reference_solution, test_DataArray):
    # Compare numpy implementation against reference solution
    experiment = reference_solution.get_experiment(
        method=method, norm=standardize, weights=use_weights
    )
    reference = experiment.get_results()

    model = EOF(standardize=standardize)
    model.fit(test_DataArray.transpose('time','x','y'), 'time')
    assert_allclose(model.singular_values(), reference['singular_values'])  #type: ignore
    assert_allclose(model.explained_variance(), reference['explained_variance'])  #type: ignore
    assert_allclose(model.explained_variance_ratio(), reference['explained_variance_ratio'])  #type: ignore
    assert_allclose(model.components().stack(loc=('x', 'y')).dropna('loc').values, reference['eofs'].T)  #type: ignore
    assert_allclose(model.scores().values, reference['pcs'].T)


def test_EOF_initialization():
    """Tests the initialization of the EOF class"""
    eof = EOF(n_modes=5, standardize=True, use_coslat=True)

    # Assert parameters are correctly stored in the _params attribute
    assert eof._params == {'n_modes': 5, 'standardize': True, 'use_coslat': True, 'use_weights': False}

    # Assert correct values are stored in the _scaling_params attribute
    assert eof._scaling_params == {'with_std': True, 'with_coslat': True, 'with_weights': False}


def test_EOF_fit(test_DataArray):
    """Tests the fit method of the EOF class"""
    dims = 'time'

    eof = EOF()
    eof.fit(test_DataArray, dims)

    # Assert that data has been preprocessed
    assert isinstance(eof.data, xr.DataArray)

    # Assert the required attributes have been set
    assert eof._total_variance is not None
    assert eof._singular_values is not None
    assert eof._explained_variance is not None
    assert eof._explained_variance_ratio is not None
    assert eof._components is not None
    assert eof._scores is not None


def test_EOF_singular_values(test_DataArray):
    """Tests the singular_values method of the EOF class"""
    dims = 'time'

    eof = EOF()
    eof.fit(test_DataArray, dims)

    # Test singular_values method
    singular_values = eof.singular_values()
    assert isinstance(singular_values, xr.DataArray)


def test_EOF_explained_variance(test_DataArray):
    """Tests the explained_variance method of the EOF class"""
    dims = 'time'

    eof = EOF()
    eof.fit(test_DataArray, dims)

    # Test explained_variance method
    explained_variance = eof.explained_variance()
    assert isinstance(explained_variance, xr.DataArray)


def test_EOF_explained_variance_ratio(test_DataArray):
    """Tests the explained_variance_ratio method of the EOF class"""
    dims = 'time'

    eof = EOF()
    eof.fit(test_DataArray, dims)

    # Test explained_variance_ratio method
    explained_variance_ratio = eof.explained_variance_ratio()
    assert isinstance(explained_variance_ratio, xr.DataArray)


def test_EOF_components(test_DataArray):
    """Tests the components method of the EOF class"""
    dims = 'time'

    eof = EOF()
    eof.fit(test_DataArray, dims)

    # Test components method
    components = eof.components()
    assert isinstance(components, xr.DataArray)


def test_EOF_scores(test_DataArray):
    """Tests the scores method of the EOF class"""
    dims = 'time'

    eof = EOF()
    eof.fit(test_DataArray, dims)

    # Test scores method
    scores = eof.scores()
    assert isinstance(scores, xr.DataArray)


def test_EOF_get_params():
    """Tests the get_params method of the EOF class"""
    eof = EOF(n_modes=5, standardize=True, use_coslat=True)

    # Test get_params method
    params = eof.get_params()
    assert isinstance(params, dict)
    assert params == {'n_modes': 5, 'standardize': True, 'use_coslat': True, 'use_weights': False}


def test_EOF_compute(test_DataArray):
    """Tests the compute method of the EOF class"""
    
    dims = 'time'
    
    dask_test_DataArray = test_DataArray.chunk({'time': 1})
    
    eof = EOF()
    eof.fit(dask_test_DataArray, dims)

   # Assert that the attributes are indeed Dask arrays before computation
    assert isinstance(eof._total_variance.data, da.Array)  #type: ignore
    assert isinstance(eof._singular_values.data, da.Array)  #type: ignore
    assert isinstance(eof._explained_variance.data, da.Array)  #type: ignore
    assert isinstance(eof._explained_variance_ratio.data, da.Array)  #type: ignore
    assert isinstance(eof._components.data, da.Array)  #type: ignore
    assert isinstance(eof._scores.data, da.Array)  #type: ignore

    # Test compute method
    eof.compute()

    # Assert the attributes are no longer Dask arrays after computation
    assert not isinstance(eof._total_variance.data, da.Array)  #type: ignore
    assert not isinstance(eof._singular_values.data, da.Array)  #type: ignore
    assert not isinstance(eof._explained_variance.data, da.Array)  #type: ignore
    assert not isinstance(eof._explained_variance_ratio.data, da.Array)  #type: ignore
    assert not isinstance(eof._components.data, da.Array)  #type: ignore
    assert not isinstance(eof._scores.data, da.Array)  #type: ignore


def test_EOF_transform(test_DataArray):
    """Test projecting new unseen data onto the components (EOFs/eigenvectors)"""

    # Create a xarray DataArray with random data
    dims = 'time'
    
    model = EOF(n_modes=2)
    model.fit(test_DataArray, dims)
    scores = model.scores()

    # Create a new xarray DataArray with random data
    new_data = test_DataArray.isel(time=range(5))

    projections = model.transform(new_data)

    # Check that the projection has the right dimensions
    assert projections.dims == scores.dims

    # Check that the projection has the right data type
    assert isinstance(projections, xr.DataArray)

    # Check that the projection has the right name
    assert projections.name == 'scores'

    # Check that the projection's data is the same as the scores
    np.testing.assert_allclose(scores.isel(time=range(5)).data, projections.data)


def test_EOF_inverse_transform(test_DataArray):
    """Test inverse_transform method in EOF class."""

    # instantiate the EOF class with necessary parameters
    dims = 'time'
    eof = EOF(n_modes=3, standardize=True)
    
    # fit the EOF model
    eof.fit(test_DataArray, dims=dims)

    # Test with scalar
    mode = 1
    reconstructed_data = eof.inverse_transform(mode)
    assert isinstance(reconstructed_data, xr.DataArray)
    
    # Test with slice
    mode = slice(1, 2)
    reconstructed_data = eof.inverse_transform(mode)
    assert isinstance(reconstructed_data, xr.DataArray)

    # Test with array of tick labels
    mode = np.array([1, 3])
    reconstructed_data = eof.inverse_transform(mode)
    assert isinstance(reconstructed_data, xr.DataArray)

    # Check that the reconstructed data has the same dimensions as the original data
    assert set(reconstructed_data.dims) == set(test_DataArray.dims)



def test_EOF_components_as_correlation(test_DataArray):
    # Setup
    dims = 'time'
    eof = EOF()
    eof.fit(test_DataArray, dims=dims)

    # Test default parameters
    corr, rejected, pvals_corr = eof.components_as_correlation()
    assert isinstance(corr, xr.DataArray), "Returned correlation is not an xarray DataArray"
    assert isinstance(rejected, xr.DataArray), "Returned rejected is not an xarray DataArray"
    assert isinstance(pvals_corr, xr.DataArray), "Returned pvals_corr is not an xarray DataArray"

    # Test alpha value
    corr, rejected, pvals_corr = eof.components_as_correlation(alpha=0.01)
    n_rejected_01 = rejected.sum('feature')
    n_rejected_05 = eof.components_as_correlation(alpha=0.05)[1].sum('feature')
    assert (n_rejected_01 <= n_rejected_05).all(), \
        "Less hypotheses should be rejected at lower significance level"

    # Test different methods
    for method in ['bonferroni', 'sidak', 'holm-sidak', 'holm', 'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky']:
        corr, rejected, pvals_corr = eof.components_as_correlation(method=method)
        assert isinstance(corr, xr.DataArray), f"Returned correlation is not an xarray DataArray for method {method}"
        assert isinstance(rejected, xr.DataArray), f"Returned rejected is not an xr.DataArray for method {method}"
        assert isinstance(pvals_corr, xr.DataArray), f"Returned pvals_corr is not an xr.DataArray for method {method}"

    # Check ValueError for invalid method
    with pytest.raises(ValueError):
        corr, rejected, pvals_corr = eof.components_as_correlation(method='invalid')

def test_ComplexEOF_fit(test_DataArray):
    """Test fitting a ComplexEOF model"""
    # Create a xarray DataArray with random data
    dims = 'time'
    
    ceof = ComplexEOF(n_modes=2)
    ceof.fit(test_DataArray, dims)

    # Check that the fit method has properly populated the attributes
    assert ceof._total_variance is not None
    assert ceof._singular_values is not None
    assert ceof._explained_variance is not None
    assert ceof._explained_variance_ratio is not None
    assert ceof._components is not None
    assert ceof._scores is not None


def test_ComplexEOF_components_amplitude(test_DataArray):
    """Test computation of components amplitude in ComplexEOF model"""
    dims = 'time'
    ceof = ComplexEOF(n_modes=2)
    ceof.fit(test_DataArray, dims)

    comp_amp = ceof.components_amplitude()
    assert comp_amp is not None
    assert (comp_amp.fillna(0) >= 0).all()  #type: ignore


def test_ComplexEOF_components_phase(test_DataArray):
    """Test computation of components phase in ComplexEOF model"""
    dims = 'time'
    ceof = ComplexEOF(n_modes=2)
    ceof.fit(test_DataArray, dims)

    comp_phase = ceof.components_phase()
    assert comp_phase is not None
    assert ((-np.pi <= comp_phase.fillna(0)) & (comp_phase.fillna(0) <= np.pi)).all()  #type: ignore


def test_ComplexEOF_scores_amplitude(test_DataArray):
    """Test computation of scores amplitude in ComplexEOF model"""
    dims = 'time'
    ceof = ComplexEOF(n_modes=2)
    ceof.fit(test_DataArray, dims)

    scores_amp = ceof.scores_amplitude()
    assert scores_amp is not None
    assert (scores_amp.fillna(0) >= 0).all()  #type: ignore


def test_ComplexEOF_scores_phase(test_DataArray):
    """Test computation of scores phase in ComplexEOF model"""
    dims = 'time'
    ceof = ComplexEOF(n_modes=2)
    ceof.fit(test_DataArray, dims)

    scores_phase = ceof.scores_phase()
    assert scores_phase is not None
    assert ((-np.pi <= scores_phase.fillna(0)) & (scores_phase.fillna(0) <= np.pi)).all()  #type: ignore
