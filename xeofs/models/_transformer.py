from typing import Union, Iterable, List

import numpy as np
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

Array = np.ndarray
ArrayList = Union[Array, List[Array]]


class _ArrayTransformer:
    '''Transform any N-D ``np.ndarray`` into a 2-D ``np.ndarray``.

    The parameter ``axis`` allows to choose which axes of the original array
    should be transformed as the first axis (rows). All other axes will be reshaped
    into the second axis (columns)

    Handles NaNs by removing all columns and/or rows which contain NaNs only.
    Additional NaNs at individual positions are not allowed and will raise
    an exception.

    '''

    def __init__(self):
        pass

    def fit(self, X : Array, axis : Union[int, Iterable[int]] = 0):
        if not isinstance(X, np.ndarray):
            raise ValueError('This interface is for `numpy.ndarray` only')
        # Convert axis to list
        if isinstance(axis, int):
            axis = list([axis])

        all_axes = np.arange(len(X.shape))
        axis_samples = axis
        axis_features = [a for a in all_axes if a not in axis_samples]
        # Be sure to include all axes
        if not np.isin((axis_samples + axis_features), all_axes).all():
            err_msg = 'One or more of axes={:} do not exist.'.format(axis)
            raise ValueError(err_msg)
        new_axes_order = axis_samples + axis_features

        shape_X = X.shape
        shape_samples = np.array(X.shape)[axis_samples]
        shape_features = np.array(X.shape)[axis_features]

        n_samples = shape_samples.prod()
        n_features = shape_features.prod()

        # Reshape into 2D array
        X = X.transpose(new_axes_order)
        shape_X_t = X.shape
        X = X.reshape(n_samples, n_features)

        # Remove "all-NaN"-columns/rows
        nan_matrix = np.isnan(X)
        idx_valid_features = ~nan_matrix.all(axis=0)
        idx_valid_samples = ~nan_matrix.all(axis=1)

        n_valid_samples = idx_valid_samples.sum()
        n_valid_features = idx_valid_features.sum()

        self.all_axes = all_axes
        self.axis_samples = axis_samples
        self.axis_features = axis_features

        self.new_axes_order = new_axes_order

        self.shape_X = shape_X
        self.shape_X_t = shape_X_t
        self.shape_samples = shape_samples
        self.shape_features = shape_features

        self.n_samples = n_samples
        self.n_features = n_features

        self.idx_valid_samples = idx_valid_samples
        self.idx_valid_features = idx_valid_features

        self.n_valid_samples = n_valid_samples
        self.n_valid_features = n_valid_features
        return self

    def transform(self, X : Array):
        # Transpose data matrix X
        try:
            X = X.transpose(self.new_axes_order)
        except ValueError:
            err_msg = 'Expected {:} dimensions, but got {:} instead.'
            err_msg = err_msg.format(len(self.all_axes), len(X.shape))
            raise ValueError(err_msg)

        shape_features = list(X.shape[len(self.axis_samples):])
        if any(self.shape_features != shape_features):
            err_msg = 'Expected feature shape {:} but got {:} instead.'
            err_msg = err_msg.format(self.shape_features, shape_features)
            raise ValueError(err_msg)

        # Reshape data matrix X
        try:
            X = X.reshape(-1, self.n_features)
        except ValueError:
            err_msg = (
                'This should not happen. Please consider raising an issue '
                'on GitHub: https://github.com/nicrie/xeofs/issues'
            )
            raise ValueError(err_msg)

        # Remove NaNs of new data
        # Remove invalid samples
        X = X[~np.isnan(X).all(axis=1)]
        # Remove invalid features from fitted data
        X = X[:, self.idx_valid_features]
        # Security check: No NaNs should remain, otherwise invalid input data
        if np.isnan(X).any():
            raise ValueError('Invalid data: contains individual NaNs')
        return X

    def transform_weights(self, weights : Array):
        shape_expected = tuple(self.shape_features)
        # Trivial case: no weighting -> weights all one
        if weights is None:
            weights = np.ones(shape_expected, dtype=int)
        try:
            shape_received = weights.shape
        except AttributeError:
            err_msg = 'weights must be of type {:}.'.format(repr(np.ndarray))
            raise TypeError(err_msg)
        # Transform weights to 1D array and remove NaNs
        if shape_expected == shape_received:
            # The dimensions order is already correct, flatten is enough
            weights = weights.flatten()
            # Remove NaN features
            weights = weights[self.idx_valid_features]
            # Consider any NaN left as zero
            return np.nan_to_num(weights)
        else:
            err_msg = 'Expected shape {:}, but got {:} instead.'
            err_msg = err_msg.format(shape_expected, shape_received)
            raise ValueError(err_msg)

    def fit_transform(
        self, X : Array,
        axis : Union[int, Iterable[int]] = 0
    ):
        return self.fit(X=X, axis=axis).transform(X)

    def back_transform(self, X : Array):
        if len(X.shape) > 2:
            raise ValueError('Data must be 2D to be back-transformed.')
        # Original shape
        Xrec = np.zeros((self.n_samples, self.n_features)) * np.nan
        # Insert data at valid locations
        valid_grid = np.ix_(self.idx_valid_samples, self.idx_valid_features)
        Xrec[valid_grid] = X
        # Inverse reshaping
        Xrec = Xrec.reshape(tuple(self.shape_X_t))
        # Inverse transposing
        return Xrec.transpose(np.argsort(self.new_axes_order))

    def back_transform_eofs(self, X : Array):
        if len(X.shape) > 2:
            raise ValueError('Data must be 2D to be back-transformed.')
        eofs = np.zeros((self.n_features, X.shape[1])) * np.nan
        eofs[self.idx_valid_features, :] = X
        return eofs.reshape(tuple(self.shape_features) + (X.shape[1],))

    def back_transform_pcs(self, X : Array):
        if len(X.shape) > 2:
            raise ValueError('Data must be 2D to be back-transformed.')
        pcs = np.zeros((self.n_samples, X.shape[1])) * np.nan
        pcs[self.idx_valid_samples, :] = X
        return pcs.reshape(tuple(self.shape_samples) + (X.shape[1],))


class _MultiArrayTransformer:
    'Transform multiple N-D ``np.ndarray`` to a single 2D ``np.ndarry``.'
    def __init__(self):
        pass

    def _convert2list(self, X):
        if np.logical_not(isinstance(X, list)):
            X = [X]
        return X

    def fit(self, X : ArrayList, axis : Union[int, Iterable[int]] = 0):
        X = self._convert2list(X)

        self.tfs = [_ArrayTransformer().fit(x, axis=axis) for x in X]

        if len(set([tf.n_valid_samples for tf in self.tfs])) > 1:
            err_msg = 'All individual arrays must have same number of samples.'
            raise ValueError(err_msg)

        self.idx_array_sep = np.cumsum([tf.n_valid_features for tf in self.tfs])
        self.axis_samples = self.tfs[0].axis_samples
        return self

    def transform(self, X : ArrayList):
        X = self._convert2list(X)

        X_transformed = [tf.transform(x) for x, tf in zip(X, self.tfs)]

        if len(set(x.shape[0] for x in X_transformed)) > 1:
            err_msg = 'All individual arrays must have same number of samples.'
            raise ValueError(err_msg)

        return np.concatenate(X_transformed, axis=1)

    def transform_weights(self, weights : ArrayList):
        weights = self._convert2list(weights)
        if len(self.tfs) != len(weights):
            weights = weights * len(self.tfs)
        if len(self.tfs) != len(weights):
            err_msg = (
                'Number of provided weights does not correspond to input data.'
            )
            raise ValueError(err_msg)

        weights = [tf.transform_weights(w) for w, tf in zip(weights, self.tfs)]
        return np.concatenate(weights, axis=0)

    def fit_transform(
        self, X : ArrayList,
        axis : Union[int, Iterable[int]] = 0
    ):
        return self.fit(X=X, axis=axis).transform(X)

    def back_transform(self, X : Array):
        Xrec = np.split(X, self.idx_array_sep[:-1], axis=1)
        Xrec = [tf.back_transform(x) for x, tf in zip(Xrec, self.tfs)]
        return Xrec

    def back_transform_eofs(self, X : Array):
        eofs = np.split(X, self.idx_array_sep[:-1], axis=0)
        eofs = [tf.back_transform_eofs(e) for e, tf in zip(eofs, self.tfs)]
        return eofs

    def back_transform_pcs(self, X : Array):
        return self.tfs[0].back_transform_pcs(X)
