from ..utils.data_types import DataArray


def total_variance(X: DataArray, dim: str) -> DataArray:
    """Compute the total variance of the centered data."""
    return (X * X.conj()).sum() / (X[dim].size - 1)
