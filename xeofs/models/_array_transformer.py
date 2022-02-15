import numpy as np
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


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
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_features_no_nan = np.sum(self.notnull_idx)
        self.shape_features = shape_features
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.shape_out_no_nan = tuple([self.n_samples, self.n_features_no_nan])
        if self.n_features_no_nan == 0:
            raise ValueError('No data to process after NaN removal.')
        return self

    def transform(self, data : np.ndarray):
        # Reshape data
        try:
            data = data.reshape((data.shape[0],) + self.shape_out[1:])
        except ValueError:
            err_msg = 'Input feature dimension is {:}, but {:} is expected.'
            err_msg = err_msg.format(data.shape[1:], self.shape_in[1:])
            raise ValueError(err_msg)
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
        if len(data.shape) != 2:
            raise ValueError('Input must be 2D.')
        data_with_nan = np.zeros([data.shape[0], self.n_features]) * np.nan
        data_with_nan[:, self.notnull_idx] = data
        return data_with_nan.reshape((-1,) + self.shape_in[1:])
