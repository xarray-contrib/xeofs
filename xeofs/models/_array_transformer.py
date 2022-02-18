from typing import Union, Iterable

import numpy as np
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


class _ArrayTransformer():
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

    def fit(self, X : np.ndarray, axis : Union[int, Iterable[int]] = 0):
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

    def transform(self, X : np.ndarray):
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

    def transform_weights(
        self,
        weights : np.ndarray
    ):
        if weights is None:
            return None
        try:
            shape_received = weights.shape
        except AttributeError:
            err_msg = 'weights must be of type {:}.'.format(repr(np.ndarray))
            raise TypeError(err_msg)
        shape_expected = tuple(self.shape_features)
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
        self, X : np.ndarray,
        axis : Union[int, Iterable[int]] = 0
    ):
        return self.fit(X=X, axis=axis).transform(X)

    def back_transform(self, X : np.ndarray):
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

    def back_transform_eofs(self, X : np.ndarray):
        if len(X.shape) > 2:
            raise ValueError('Data must be 2D to be back-transformed.')
        eofs = np.zeros((self.n_features, X.shape[1])) * np.nan
        eofs[self.idx_valid_features, :] = X
        return eofs.reshape(tuple(self.shape_features) + (X.shape[1],))

    def back_transform_pcs(self, X : np.ndarray):
        if len(X.shape) > 2:
            raise ValueError('Data must be 2D to be back-transformed.')
        pcs = np.zeros((self.n_samples, X.shape[1])) * np.nan
        pcs[self.idx_valid_samples, :] = X
        return pcs.reshape(tuple(self.shape_samples) + (X.shape[1],))
