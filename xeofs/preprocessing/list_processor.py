from typing import List, TypeVar, Generic, Type, Dict, Any
from typing_extensions import Self

from .dimension_renamer import DimensionRenamer
from .scaler import Scaler
from .sanitizer import Sanitizer
from .multi_index_converter import MultiIndexConverter
from .stacker import Stacker
from ..utils.data_types import (
    Data,
    DataVar,
    DataVarBound,
    DataArray,
    DataSet,
    Dims,
    DimsList,
)

T = TypeVar(
    "T",
    bound=(DimensionRenamer | Scaler | MultiIndexConverter | Stacker | Sanitizer),
)


class GenericListTransformer(Generic[T]):
    """Apply a Transformer to each of the elements of a list.

    Parameters
    ----------
    transformer: Transformer
        Transformer class to apply to list elements.
    kwargs: dict
        Keyword arguments for the transformer.
    """

    def __init__(self, transformer: Type[T], **kwargs):
        self.transformer_class = transformer
        self.transformers: List[T] = []
        self.init_kwargs = kwargs

    def fit(
        self,
        X: List[DataVar],
        sample_dims: Dims,
        feature_dims: DimsList,
        iter_kwargs: Dict[str, List[Any]] = {},
    ) -> Self:
        """Fit transformer to each data element in the list.

        Parameters
        ----------
        X: List[Data]
            List of data elements.
        sample_dims: Dims
            Sample dimensions.
        feature_dims: DimsList
            Feature dimensions.
        iter_kwargs: Dict[str, List[Any]]
            Keyword arguments for the transformer that should be iterated over.

        """
        self._sample_dims = sample_dims
        self._feature_dims = feature_dims
        self._iter_kwargs = iter_kwargs

        for i, x in enumerate(X):
            # Add transformer specific keyword arguments
            # For iterable kwargs, use the i-th element of the iterable
            kwargs = {k: v[i] for k, v in self._iter_kwargs.items()}
            proc: T = self.transformer_class(**self.init_kwargs)
            proc.fit(x, sample_dims, feature_dims[i], **kwargs)
            self.transformers.append(proc)
        return self

    def transform(self, X: List[Data]) -> List[Data]:
        X_transformed: List[Data] = []
        for x, proc in zip(X, self.transformers):
            X_transformed.append(proc.transform(x))  #  type: ignore
        return X_transformed

    def fit_transform(
        self,
        X: List[Data],
        sample_dims: Dims,
        feature_dims: DimsList,
        iter_kwargs: Dict[str, List[Any]] = {},
    ) -> List[Data]:
        return self.fit(X, sample_dims, feature_dims, iter_kwargs).transform(X)  # type: ignore

    def inverse_transform_data(self, X: List[Data]) -> List[Data]:
        X_inverse_transformed: List[Data] = []
        for x, proc in zip(X, self.transformers):
            x_inv_trans = proc.inverse_transform_data(x)  #  type: ignore
            X_inverse_transformed.append(x_inv_trans)
        return X_inverse_transformed

    def inverse_transform_components(self, X: List[Data]) -> List[Data]:
        X_inverse_transformed: List[Data] = []
        for x, proc in zip(X, self.transformers):
            x_inv_trans = proc.inverse_transform_components(x)  #  type: ignore
            X_inverse_transformed.append(x_inv_trans)
        return X_inverse_transformed

    def inverse_transform_scores(self, X: DataArray) -> DataArray:
        return self.transformers[0].inverse_transform_scores(X)

    def inverse_transform_scores_unseen(self, X: DataArray) -> DataArray:
        return self.transformers[0].inverse_transform_scores_unseen(X)
