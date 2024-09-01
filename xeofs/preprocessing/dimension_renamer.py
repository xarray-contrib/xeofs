from typing_extensions import Self

from ..utils.data_types import Data, DataArray, DataVarBound, Dims
from .transformer import Transformer


class DimensionRenamer(Transformer):
    """Rename dimensions of a DataArray or Dataset.

    Parameters
    ----------
    base: str
        Base string for the new dimension names.
    start: int
        Start index for the new dimension names.

    """

    def __init__(self, base="dim", start=0):
        super().__init__()
        self.base = base
        self.start = start
        self.dim_mapping = {}

    def get_serialization_attrs(self) -> dict:
        return dict(
            dim_mapping=self.dim_mapping,
        )

    def fit(self, X: Data, sample_dims: Dims, feature_dims: Dims, **kwargs) -> Self:
        self.sample_dims_before = sample_dims
        self.feature_dims_before = feature_dims

        self.dim_mapping = {
            dim: f"{self.base}{i}" for i, dim in enumerate(X.dims, start=self.start)
        }

        self.sample_dims_after: Dims = tuple(
            [self.dim_mapping[dim] for dim in self.sample_dims_before]
        )
        self.feature_dims_after: Dims = tuple(
            [self.dim_mapping[dim] for dim in self.feature_dims_before]
        )

        return self

    def transform(self, X: DataVarBound) -> DataVarBound:
        try:
            return X.rename(self.dim_mapping)
        except ValueError:
            raise ValueError("Cannot transform data. Dimensions are different.")

    def _inverse_transform(self, X: DataVarBound) -> DataVarBound:
        given_dims = set(X.dims)
        expected_dims = set(self.dim_mapping.values())
        dims = given_dims.intersection(expected_dims)
        return X.rename({v: k for k, v in self.dim_mapping.items() if v in dims})

    def inverse_transform_data(self, X: DataVarBound) -> DataVarBound:
        return self._inverse_transform(X)

    def inverse_transform_components(self, X: DataVarBound) -> DataVarBound:
        return self._inverse_transform(X)

    def inverse_transform_scores(self, X: DataArray) -> DataArray:
        return self._inverse_transform(X)

    def inverse_transform_scores_unseen(self, X: DataArray) -> DataArray:
        return self._inverse_transform(X)
