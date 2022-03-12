import numpy as np
import pytest
import warnings
from numpy.testing import assert_array_equal
from xeofs.models._transformer import _ArrayTransformer, _MultiArrayTransformer

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


@pytest.mark.parametrize('input_shape, axis', [
    ((100, 10), 0),
    ((100, 10), 1),
    ((100, 10, 10), 0),
    ((100, 10, 10), [1, 2]),
])
def test_output_shape(input_shape, axis):
    # Output array is always 2D
    rng = np.random.default_rng(7)
    arr_in = rng.standard_normal(input_shape)
    tf = _ArrayTransformer()
    arr_out = tf.fit_transform(arr_in)
    assert len(arr_out.shape) == 2


@pytest.mark.parametrize('input_shape, axis', [
    ((10, 4) , 2),
    ((10, 4) , [0, 2]),
])
def test_invalid_axis(input_shape, axis):
    # Invalid axis argument
    rng = np.random.default_rng(7)
    arr_in = rng.standard_normal(input_shape)
    tf = _ArrayTransformer()
    with pytest.raises(Exception):
        _ = tf.fit(arr_in, axis=axis)


@pytest.mark.parametrize('shape', [
    (10, 5, 5),
    (10, 5, 4, 2),
])
def test_invalid_transform(shape):
    # Invalid data shape
    rng = np.random.default_rng(7)
    Xfit = rng.standard_normal((10, 5, 4))
    X = rng.standard_normal(shape)
    tf = _ArrayTransformer()
    _ = tf.fit(Xfit)
    with pytest.raises(Exception):
        _ = tf.transform(X)


def test_valid_transform_with_nan():
    # Valid data with NaNs has no NaNs after transform
    rng = np.random.default_rng(7)
    X = rng.standard_normal((10, 5, 4))
    X[:, 0, 0] = np.nan
    tf = _ArrayTransformer()
    transformed = tf.fit_transform(X)
    assert ~np.isnan(transformed).any()


def test_invalid_transform_with_nan():
    # Invalid data with individual nan
    rng = np.random.default_rng(7)
    Xfit = rng.standard_normal((10, 5, 4))
    X = rng.standard_normal(Xfit.shape)
    X[0, 0, 0] = np.nan
    tf = _ArrayTransformer()
    _ = tf.fit(Xfit)
    with pytest.raises(Exception):
        _ = tf.transform(X)


@pytest.mark.parametrize('weights', [
    np.array([1, 2, 3]),
    None,
])
def test_transform_weight_data_types(weights):
    # Output data type is always np.ndarray
    rng = np.random.default_rng(7)
    X = rng.standard_normal((5, 3))
    tf = _ArrayTransformer()
    _ = tf.fit(X)
    transformed = tf.transform_weights(weights)
    assert isinstance(transformed, np.ndarray)


@pytest.mark.parametrize('weights', [
    [1, 2, 3],
    4,
    (1, 2, 3),
])
def test_invalid_transform_weight_data_types(weights):
    # Data types other than np.ndarray or None raise an exception
    rng = np.random.default_rng(7)
    X = rng.standard_normal((5, 3))
    tf = _ArrayTransformer()
    _ = tf.fit(X)
    with pytest.raises(Exception):
        _ = tf.transform_weights(weights)


@pytest.mark.parametrize('data_shape, weight_shape, axis', [
    ((10, 4), (3), 0),
    ((10, 4, 2), (3, 2), 0),
    ((10, 4, 2), (9), [1, 2]),
])
def test_invalid_transform_weight_shapes(data_shape, weight_shape, axis):
    # Weight shapes and shape of features are different
    rng = np.random.default_rng(7)
    X = rng.standard_normal(data_shape)
    rng = np.random.default_rng(8)
    weights = rng.standard_normal(weight_shape)
    tf = _ArrayTransformer()
    _ = tf.fit(X, axis=axis)
    with pytest.raises(Exception):
        _ = tf.transform_weights(weights)


@pytest.mark.parametrize('shape, axis', [
    ((10, 2), 0),
    ((10, 2, 5), 1),
    ((10, 2, 5), [1, 2]),
])
def test_back_transformation(shape, axis):
    # Transform and back transform yields initial data
    rng = np.random.default_rng(7)
    X = rng.standard_normal(shape)
    tf = _ArrayTransformer()
    transformed = tf.fit_transform(X)
    Xrec = tf.back_transform(transformed)
    _ = tf.back_transform_eofs(transformed.T)
    _ = tf.back_transform_pcs(transformed)
    assert_array_equal(X, Xrec)


@pytest.mark.parametrize('shape, axis', [
    ((10, 2), 0),
    ((10, 2, 5), 1),
    ((10, 2, 5), [1, 2]),
])
def test_invalid_back_transformation(shape, axis):
    # Back transforming of 3D array is invalid
    rng = np.random.default_rng(7)
    X = rng.standard_normal(shape)
    tf = _ArrayTransformer()
    transformed = tf.fit_transform(X)
    with pytest.raises(Exception):
        _ = tf.back_transform(transformed[:, np.newaxis])
    with pytest.raises(Exception):
        _ = tf.back_transform_eofs(transformed[:, np.newaxis])
    with pytest.raises(Exception):
        _ = tf.back_transform_pcs(transformed[:, np.newaxis])


def test_invalid_multi_transform():
    # Differet number of samples raises error
    shape1 = (10, 5, 4)
    shape2 = (10, 8, 6)
    rng = np.random.default_rng(7)
    X1 = rng.standard_normal(shape1)
    X2 = rng.standard_normal(shape2)
    tf = _MultiArrayTransformer()
    tf.fit([X1, X2])

    new_shape1 = (3, 5, 4)
    new_shape2 = (4, 8, 6)
    rng = np.random.default_rng(7)
    new_X1 = rng.standard_normal(new_shape1)
    new_X2 = rng.standard_normal(new_shape2)
    with pytest.raises(Exception):
        tf.fit([new_X1, new_X2])
    with pytest.raises(Exception):
        tf.transform([new_X1, new_X2])


def test_invalid_multi_transform_weights():
    # Incorrect number of features raises error
    shape1 = (10, 5, 4)
    shape2 = (10, 8, 6)
    rng = np.random.default_rng(7)
    X1 = rng.standard_normal(shape1)
    X2 = rng.standard_normal(shape2)
    tf = _MultiArrayTransformer()
    tf.fit([X1, X2])

    weights_incorrect_shape1 = (5, 3)
    weights_shape2 = (8, 6)
    rng = np.random.default_rng(7)
    weights1 = rng.standard_normal(weights_incorrect_shape1)
    weights2 = rng.standard_normal(weights_shape2)
    with pytest.raises(Exception):
        tf.transform_weights([weights1, weights2])
