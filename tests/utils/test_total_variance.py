import numpy as np
import xarray as xr

from xeofs.utils.xarray_utils import total_variance


def test_total_variance(mock_data_array):
    """Test the total_variance function."""
    arr = mock_data_array.copy().values
    # Compute total variance using numpy by hand
    tot_var_ref = np.sum(np.var(arr, axis=0, ddof=1))

    da = mock_data_array.stack(sample=("time",), feature=("lat", "lon"))
    da -= da.mean(dim="sample")
    tot_var = total_variance(da, dim="sample")
    assert isinstance(tot_var, xr.DataArray)
    assert tot_var.dims == ()
    assert tot_var.shape == ()
    assert tot_var.dtype == np.float64
    assert np.allclose(
        tot_var, tot_var_ref
    ), "Total variance computed by hand does not match."
