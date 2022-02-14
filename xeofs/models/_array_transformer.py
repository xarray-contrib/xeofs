import numpy as np


class _ArrayTransformer():
    '''Transform any N-D `numpy.ndarray` into a 2D ndarray keeping the
    first axis fixed.

    Handles NaNs by removing all columns containing at least one NaN.

    '''

    def __init__(self):
        pass

    def fit(self, data : np.ndarray):
        shape_in = data.shape
        shape_features = data.shape[1:]
        n_samples = data.shape[0]
        n_features = np.product(shape_features)
        shape_out = tuple([n_samples, n_features])

        if any([dim <= 1 for dim in shape_out]):
            raise ValueError('Final input data must be 2D but is 1D.')

        nan_idx = np.isnan(data.reshape(shape_out)).any(axis=0)
        self.notnull_idx = np.logical_not(nan_idx)
        if self.notnull_idx.sum() == 0:
            raise ValueError('No data to process after NaN removal.')
        self.n_samples = n_samples
        self.n_features = n_features
        self.shape_features = shape_features
        self.shape_out = shape_out
        self.shape_in = shape_in
        return self

    def transform(self, data : np.ndarray):
        # Reshape data
        try:
            data = data.reshape((data.shape[0],) + self.shape_out[1:])
        except ValueError:
            err_msg = 'Input feature dimension is {:}, but {:} is expected.'
            err_msg = err_msg.format(data.shape[1:], self.shape_in[1:])
        # Remove NaNs
        data = data[:, self.notnull_idx]
        if np.isnan(data).any():
            err_msg = 'NaNs must be at same location as fitted data.'
            raise ValueError(err_msg)
        else:
            return data

    def fit_transform(self, data : np.ndarray):
        return self.fit(data).transform(data)

    def back_transform(self, data : np.ndarray):
        n_vars_in = len(self.notnull_idx)
        data_with_nan = np.zeros([data.shape[0], n_vars_in]) * np.nan
        try:
            data_with_nan[:, self.notnull_idx] = data
        except ValueError as err:
            raise err
        return data_with_nan.reshape((-1,) + self.shape_in[1:])
