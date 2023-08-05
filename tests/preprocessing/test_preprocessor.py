import pytest
import numpy as np
import xarray as xr

from xeofs.preprocessing.preprocessor import Preprocessor


@pytest.mark.parametrize(
    "with_std, with_coslat, with_weights",
    [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, True, True),
        (False, True, False),
        (False, False, True),
        (False, False, False),
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, True, True),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ],
)
def test_init_params(with_std, with_coslat, with_weights):
    prep = Preprocessor(
        with_std=with_std, with_coslat=with_coslat, with_weights=with_weights
    )

    assert hasattr(prep, "_params")
    assert prep._params["with_std"] == with_std
    assert prep._params["with_coslat"] == with_coslat
    assert prep._params["with_weights"] == with_weights


@pytest.mark.parametrize(
    "with_std, with_coslat, with_weights",
    [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, True, True),
        (False, True, False),
        (False, False, True),
        (False, False, False),
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, True, True),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ],
)
def test_fit(with_std, with_coslat, with_weights, mock_data_array):
    """fit method should not be implemented."""
    prep = Preprocessor(
        with_std=with_std, with_coslat=with_coslat, with_weights=with_weights
    )

    with pytest.raises(NotImplementedError):
        prep.fit(mock_data_array, dim="time")


@pytest.mark.parametrize(
    "with_std, with_coslat, with_weights",
    [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, True, True),
        (False, True, False),
        (False, False, True),
        (False, False, False),
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, True, True),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ],
)
def test_fit_transform(with_std, with_coslat, with_weights, mock_data_array):
    """fit method should not be implemented."""
    prep = Preprocessor(
        with_std=with_std, with_coslat=with_coslat, with_weights=with_weights
    )

    weights = None
    if with_weights:
        weights = mock_data_array.mean("time").copy()
        weights[:] = 1.0

    data_trans = prep.fit_transform(mock_data_array, weights=weights, dim="time")

    assert hasattr(prep, "scaler")
    assert hasattr(prep, "stacker")

    # Transformed data is centered
    assert np.isclose(data_trans.mean("sample"), 0).all()
