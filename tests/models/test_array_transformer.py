import numpy as np
import pytest
import warnings

from xeofs.models._array_transformer import _ArrayTransformer

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


@pytest.mark.parametrize('input_shape', [
    (100, 10),
    (100, 10, 5),
    (2, 2, 1, 4)
])
def test_output_shape(input_shape):
    # Output array is always 2D with fixed first axis
    rng = np.random.default_rng(7)
    arr_in = rng.standard_normal(input_shape)
    tf = _ArrayTransformer()
    arr_out = tf.fit_transform(arr_in)
    assert (len(arr_out.shape) == 2) & (arr_out.shape[0] == input_shape[0])


@pytest.mark.parametrize('input_shape', [
    (1),
    (10),
    (1, 10),
    (10, 1),
])
def test_invalid_input_shape(input_shape):
    # Invalid data shape (forms of 1D arrays)
    rng = np.random.default_rng(7)
    arr_in = rng.standard_normal(input_shape)
    tf = _ArrayTransformer()
    with pytest.raises(Exception):
        _ = tf.fit_transform(arr_in)


def test_valid_data_with_nan():
    # Valid input data containing NaNs
    m, n = [10, 5]
    rng = np.random.default_rng(7)
    data_with_nan = rng.standard_normal((m, n))
    data_with_nan[2, 3] = np.nan
    tf = _ArrayTransformer()
    data_out = tf.fit_transform(data_with_nan)
    assert (len(data_out.shape) == 2) & (data_out.shape[0] == m)


def test_invalid_data_with_nan():
    # Invalid input data containing NaNs
    rng = np.random.default_rng(7)
    data_with_nan = rng.standard_normal((10, 5))
    data_with_nan[0, :] = np.nan
    tf = _ArrayTransformer()
    with pytest.raises(Exception):
        _ = tf.fit_transform(data_with_nan)


@pytest.mark.parametrize(
    'data_shape, new_data_shape, nan_location',
    [
        ((10, 5), (10, 5), 0),
        ((10, 5), (1, 5), 0),
        ((10, 5, 4), (2, 5, 4), 0),
    ]
)
def test_valid_transform(data_shape, new_data_shape, nan_location):
    # Transformed shape matches fitted output shape.
    rng = np.random.default_rng(7)
    data_with_nan = rng.standard_normal(data_shape)
    data_with_nan[:, 0] = np.nan

    rng = np.random.default_rng(8)
    new_data = rng.standard_normal(new_data_shape)
    new_data[:, nan_location] = np.nan
    tf = _ArrayTransformer()
    tf.fit(data_with_nan)
    data_transformed = tf.transform(new_data)

    assert tf.shape_out_no_nan[1:] == data_transformed.shape[1:]


@pytest.mark.parametrize(
    'data_shape, new_data_shape, nan_location',
    [
        ((10, 5), (10, 5, 2), 0),
        ((10, 5), (10, 6), 0),
        ((10, 5), (10, 5), 1),
    ]
)
def test_invalid_transform(data_shape, new_data_shape, nan_location):
    rng = np.random.default_rng(7)
    data_with_nan = rng.standard_normal(data_shape)
    data_with_nan[:, 0] = np.nan

    rng = np.random.default_rng(8)
    new_data = rng.standard_normal(new_data_shape)
    new_data[:, nan_location] = np.nan
    tf = _ArrayTransformer()
    tf.fit(data_with_nan)
    with pytest.raises(Exception):
        _ = tf.transform(new_data)


@pytest.mark.parametrize(
    'data_shape, new_data_shape',
    [
        ((10, 5), (1, 4)),
        ((10, 5, 3), (2, 12)),
    ]
)
def test_valid_back_transformation(data_shape, new_data_shape):
    rng = np.random.default_rng(7)
    data_with_nan = rng.standard_normal(data_shape)
    data_with_nan[0, 0] = np.nan

    rng = np.random.default_rng(8)
    new_data = rng.standard_normal(new_data_shape)

    tf = _ArrayTransformer()
    tf.fit(data_with_nan)
    back_transformed = tf.back_transform(new_data)
    assert data_shape[1:] == back_transformed.shape[1:]


@pytest.mark.parametrize(
    'data_shape, new_data_shape',
    [
        ((10, 5), (4)),
        ((10, 5, 3), (2, 13)),
    ]
)
def test_invalid_back_transformation(data_shape, new_data_shape):
    rng = np.random.default_rng(7)
    data_with_nan = rng.standard_normal(data_shape)
    data_with_nan[0, 0] = np.nan

    rng = np.random.default_rng(8)
    new_data = rng.standard_normal(new_data_shape)

    tf = _ArrayTransformer()
    tf.fit(data_with_nan)
    with pytest.raises(Exception):
        _ = tf.back_transform(new_data)
