import pytest
import numpy as np
import xarray as xr
from dask.array import Array as DaskArray  # type: ignore


from xeofs.data_container.eof_data_container import ComplexEOFDataContainer


def test_init():
    """Test the initialization of the ComplexEOFDataContainer."""
    container = ComplexEOFDataContainer()
    assert container._input_data is None
    assert container._components is None
    assert container._scores is None
    assert container._explained_variance is None
    assert container._total_variance is None
    assert container._idx_modes_sorted is None


def test_set_data(
    sample_input_data,
    sample_components,
    sample_scores,
    sample_exp_var,
    sample_total_variance,
    sample_idx_modes_sorted,
):
    """Test the set_data() method."""

    container = ComplexEOFDataContainer()
    container.set_data(
        sample_input_data,
        sample_components,
        sample_scores,
        sample_exp_var,
        sample_total_variance,
        sample_idx_modes_sorted,
    )
    total_variance = sample_exp_var.sum()
    idx_modes_sorted = sample_exp_var.argsort()[::-1]
    container.set_data(
        input_data=sample_input_data,
        components=sample_components,
        scores=sample_scores,
        explained_variance=sample_exp_var,
        total_variance=total_variance,
        idx_modes_sorted=idx_modes_sorted,
    )
    assert container._input_data is sample_input_data
    assert container._components is sample_components
    assert container._scores is sample_scores
    assert container._explained_variance is sample_exp_var
    assert container._total_variance is total_variance
    assert container._idx_modes_sorted is idx_modes_sorted


def test_no_data():
    """Test the data accessors without data."""
    container = ComplexEOFDataContainer()
    with pytest.raises(ValueError):
        container.input_data
    with pytest.raises(ValueError):
        container.components
    with pytest.raises(ValueError):
        container.scores
    with pytest.raises(ValueError):
        container.explained_variance
    with pytest.raises(ValueError):
        container.total_variance
    with pytest.raises(ValueError):
        container.idx_modes_sorted
    with pytest.raises(ValueError):
        container.set_attrs({"test": 1})
    with pytest.raises(ValueError):
        container.compute()


def test_set_attrs(sample_input_data, sample_components, sample_scores, sample_exp_var):
    """Test the set_attrs() method."""
    total_variance = sample_exp_var.chunk({"mode": 2}).sum()
    idx_modes_sorted = sample_exp_var.argsort()[::-1]
    container = ComplexEOFDataContainer()
    container.set_data(
        sample_input_data,
        sample_components,
        sample_scores,
        sample_exp_var,
        total_variance,
        idx_modes_sorted,
    )
    container.set_attrs({"test": 1})
    assert container.components.attrs["test"] == 1
    assert container.scores.attrs["test"] == 1
    assert container.explained_variance.attrs["test"] == 1
    assert container.explained_variance_ratio.attrs["test"] == 1
    assert container.singular_values.attrs["test"] == 1
    assert container.total_variance.attrs["test"] == 1
    assert container.idx_modes_sorted.attrs["test"] == 1


def test_compute(sample_input_data, sample_components, sample_scores, sample_exp_var):
    """Check that dask arrays are computed correctly."""
    total_variance = sample_exp_var.chunk({"mode": 2}).sum()
    idx_modes_sorted = sample_exp_var.argsort()[::-1]
    container = ComplexEOFDataContainer()
    container.set_data(
        sample_input_data.chunk({"sample": 2}),
        sample_components.chunk({"feature": 2}),
        sample_scores.chunk({"sample": 2}),
        sample_exp_var.chunk({"mode": 2}),
        total_variance,
        idx_modes_sorted,
    )
    # The components and scores are dask arrays
    assert isinstance(container.input_data.data, DaskArray)
    assert isinstance(container.components.data, DaskArray)
    assert isinstance(container.scores.data, DaskArray)
    assert isinstance(container.explained_variance.data, DaskArray)
    assert isinstance(container.total_variance.data, DaskArray)

    container.compute()

    # The components and scores are computed correctly
    assert isinstance(
        container.input_data.data, DaskArray
    ), "input_data should still be a dask array"
    assert isinstance(container.components.data, np.ndarray)
    assert isinstance(container.scores.data, np.ndarray)
    assert isinstance(container.explained_variance.data, np.ndarray)
    assert isinstance(container.total_variance.data, np.ndarray)
