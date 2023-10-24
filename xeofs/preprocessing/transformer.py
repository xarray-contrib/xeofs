from typing import Optional
from typing_extensions import Self
from abc import abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin

from ..utils.data_types import Dims, DataVar, DataArray, DataSet, Data, DataVarBound


class Transformer(BaseEstimator, TransformerMixin):
    """
    Abstract base class to transform an xarray DataArray/Dataset.

    """

    def __init__(
        self,
        sample_name: str = "sample",
        feature_name: str = "feature",
    ):
        self.sample_name = sample_name
        self.feature_name = feature_name

    @abstractmethod
    def fit(
        self,
        X: Data,
        sample_dims: Optional[Dims] = None,
        feature_dims: Optional[Dims] = None,
        **kwargs
    ) -> Self:
        """Fit transformer to data.

        Parameters:
        -------------
        X: xr.DataArray | xr.Dataset
            Input data.
        sample_dims: Sequence[Hashable], optional
            Sample dimensions.
        feature_dims: Sequence[Hashable], optional
            Feature dimensions.
        """
        pass

    @abstractmethod
    def transform(self, X: Data) -> Data:
        return X

    def fit_transform(
        self,
        X: Data,
        sample_dims: Optional[Dims] = None,
        feature_dims: Optional[Dims] = None,
        **kwargs
    ) -> Data:
        return self.fit(X, sample_dims, feature_dims, **kwargs).transform(X)

    @abstractmethod
    def inverse_transform_data(self, X: Data) -> Data:
        return X

    @abstractmethod
    def inverse_transform_components(self, X: Data) -> Data:
        return X

    @abstractmethod
    def inverse_transform_scores(self, X: DataArray) -> DataArray:
        return X
