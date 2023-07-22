import numpy as np

from ..data_container.eof_data_container import EOFDataContainer
from ..utils.data_types import DataArray


class EOFBootstrapperDataContainer(EOFDataContainer):
    '''Container that holds the data related to a Bootstrapper EOF model.
     
    '''

    @staticmethod
    def _verify_dims(da: DataArray, dims: tuple):
        '''Verify that the dimensions of the data are correct.'''
        # Bootstrapper EOFs have an additional dimension for the bootstrap
        expected_dims = dims
        given_dims = da.dims

        # In the case of the input data, the dimensions are ('sample', 'feature')
        # Otherwise, the data should have an additional dimension for the bootstrap `n`
        has_input_data_dims = set(given_dims) == set(('sample', 'feature'))
        if not has_input_data_dims:
            expected_dims = ('n',) + dims

        dims_are_equal = set(given_dims) == set(expected_dims)
        if not dims_are_equal:
            raise ValueError(f'The data must have dimensions {expected_dims}.')