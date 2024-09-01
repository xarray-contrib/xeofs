import numpy as np
from typing_extensions import Self

from ..utils.data_types import (
    Data,
    DataArray,
    Dims,
    DimsList,
)
from ..utils.xarray_utils import (
    _check_parameter_number,
    convert_to_list,
    get_dims,
    process_parameter,
    unwrap_singleton_list,
)
from .concatenator import Concatenator
from .dimension_renamer import DimensionRenamer
from .list_processor import GenericListTransformer
from .multi_index_converter import MultiIndexConverter
from .sanitizer import Sanitizer
from .scaler import Scaler
from .stacker import Stacker
from .transformer import Transformer

try:
    from xarray.core.datatree import DataTree
except ImportError:
    from datatree import DataTree


def extract_new_dim_names(X: list[DimensionRenamer]) -> tuple[Dims, DimsList]:
    """Extract the new dimension names from a list of DimensionRenamer objects.

    Parameters
    ----------
    X : list of DimensionRenamer
        list of DimensionRenamer objects.

    Returns
    -------
    Dims
        Sample dimensions
    DimsList
        Feature dimenions

    """
    new_sample_dims = []
    new_feature_dims: DimsList = []
    for x in X:
        new_sample_dims.append(x.sample_dims_after)
        new_feature_dims.append(x.feature_dims_after)
    new_sample_dims: Dims = tuple(np.unique(np.asarray(new_sample_dims)).tolist())
    return new_sample_dims, new_feature_dims


class Preprocessor(Transformer):
    """Preprocess xarray objects (DataArray, Dataset).

    Preprocessing includes
        (i) Feature-wise scaling (e.g. removing mean, dividing by standard deviation, applying (latitude) weights
        (ii) Renaming dimensions (to avoid conflicts with sample and feature dimensions)
        (iii) Converting MultiIndexes to regular Indexes (MultiIndexes cannot be stacked)
        (iv) Stacking the data into 2D DataArray
        (v) Converting MultiIndexes introduced by stacking into regular Indexes
        (vi) Removing NaNs
        (vii) Concatenating the 2D DataArrays into one 2D DataArray

    Parameters
    ----------
    sample_name : str, default="sample"
        Name of the sample dimension.
    feature_name : str, default="feature"
        Name of the feature dimension.
    with_center : bool, default=True
        If True, the data is centered by subtracting the mean.
    with_std : bool, default=True
        If True, the data is divided by the standard deviation.
    with_coslat : bool, default=False
        If True, the data is multiplied by the square root of cosine of latitude weights.
    with_weights : bool, default=False
        If True, the data is multiplied by additional user-defined weights.
    return_list : bool, default=True
        If True, inverse_transform methods returns always a list of DataArray(s).
        If False, the output is returned as a single DataArray if possible.
    check_nans : bool, default=True
        If True, remove full-dimensional NaN features from the data, check to ensure
        that NaN features match the original fit data during transform, and check
        for isolated NaNs. Note: this forces eager computation of dask arrays.
        If False, skip all NaN checks. In this case, NaNs should be explicitly removed
        or filled prior to fitting, or SVD will fail.

    """

    def __init__(
        self,
        sample_name: str = "sample",
        feature_name: str = "feature",
        with_center: bool = True,
        with_std: bool = False,
        with_coslat: bool = False,
        return_list: bool = True,
        check_nans: bool = True,
        compute: bool = True,
    ):
        # Set parameters
        self.sample_name = sample_name
        self.feature_name = feature_name
        self.with_center = with_center
        self.with_std = with_std
        self.with_coslat = with_coslat
        self.return_list = return_list
        self.check_nans = check_nans
        self.compute = compute

        self.n_data = None

        dim_names_as_kwargs = {
            "sample_name": self.sample_name,
            "feature_name": self.feature_name,
        }

        # Initialize transformers
        # 1 | Center, scale and weigh the data
        scaler_kwargs = {
            "with_center": self.with_center,
            "with_std": self.with_std,
            "with_coslat": self.with_coslat,
        }
        self.scaler = GenericListTransformer(
            Scaler, compute=self.compute, **scaler_kwargs
        )
        # 2 | Rename dimensions
        self.renamer = GenericListTransformer(DimensionRenamer)
        # 3 | Convert MultiIndexes (before stacking)
        self.preconverter = GenericListTransformer(MultiIndexConverter)
        # 4 | Stack the data to 2D DataArray
        self.stacker = GenericListTransformer(Stacker, **dim_names_as_kwargs)
        # 5 | Convert MultiIndexes (after stacking)
        self.postconverter = GenericListTransformer(MultiIndexConverter)
        # 6 | Remove NaNs
        self.sanitizer = GenericListTransformer(
            Sanitizer, check_nans=self.check_nans, **dim_names_as_kwargs
        )
        # 7 | Concatenate into one 2D DataArray
        self.concatenator = Concatenator(**dim_names_as_kwargs)

    def get_serialization_attrs(self) -> dict:
        return dict(n_data=self.n_data)

    def transformer_types(self):
        """Ordered list of transformer operations."""
        return dict(
            scaler=Scaler,
            renamer=DimensionRenamer,
            preconverter=MultiIndexConverter,
            stacker=Stacker,
            postconverter=MultiIndexConverter,
            sanitizer=Sanitizer,
            concatenator=Concatenator,
        )

    def get_transformers(self, inverse: bool = False):
        transformers = [getattr(self, t) for t in self.transformer_types().keys()]
        if inverse:
            transformers = transformers[::-1]
        return transformers

    def fit(
        self,
        X: list[Data] | Data,
        sample_dims: Dims,
        weights: list[Data] | Data | None = None,
    ) -> Self:
        """Fit the preprocessor to the data.

        Parameters
        ----------
        X : xarray objects or list of xarray objects
            Input data.
        sample_dims : tuple of str
            Sample dimensions.
        weights : xr.DataArray or list of xr.DataArray, optional
            Weights to be applied to the data.

        Returns
        -------
        self : Preprocessor
            The fitted preprocessor.

        """
        self, X = self._fit_algorithm(X, sample_dims, weights)
        return self

    def _fit_algorithm(
        self,
        X: list[Data] | Data,
        sample_dims: Dims,
        weights: list[Data] | Data | None = None,
    ) -> tuple[Self, Data]:
        self._set_return_list(X)
        X = convert_to_list(X)
        self.n_data = len(X)
        sample_dims, feature_dims = get_dims(X, sample_dims)

        # For each DataArray a list of feature dimensions must be provided
        _check_parameter_number("feature_dims", feature_dims, self.n_data)

        # Ensure that weights are provided as a list
        weights = process_parameter("weights", weights, None, self.n_data)

        # 1 | Center, scale and weigh the data
        scaler_iterkwargs = {"weights": weights}
        X = self.scaler.fit_transform(
            X=X,
            sample_dims=sample_dims,
            feature_dims=feature_dims,
            iter_kwargs=scaler_iterkwargs,
        )
        # 2 | Rename dimensions
        X = self.renamer.fit_transform(X, sample_dims, feature_dims)
        sample_dims, feature_dims = extract_new_dim_names(self.renamer.transformers)
        # 3 | Convert MultiIndexes (before stacking)
        X = self.preconverter.fit_transform(X, sample_dims, feature_dims)
        # 4 | Stack the data to 2D DataArray
        X = self.stacker.fit_transform(X, sample_dims, feature_dims)
        # 5 | Convert MultiIndexes (after stacking)
        X = self.postconverter.fit_transform(X, sample_dims, feature_dims)
        # 6 | Remove NaNs
        X = self.sanitizer.fit_transform(X, sample_dims, feature_dims)
        # 7 | Concatenate into one 2D DataArray
        X = self.concatenator.fit_transform(X)  # type: ignore

        return self, X

    def transform(self, X: list[Data] | Data) -> DataArray:
        """Transform the data.

        Parameters
        ----------
        X : xarray objects or list of xarray objects
            Input data.

        Returns
        -------
        xr.DataArray
            The transformed data.

        """
        X = convert_to_list(X)

        if len(X) != self.n_data:
            raise ValueError(
                f"number of data objects passed should match number of data objects used for fitting"
                f"len(data objects)={len(X)} and "
                f"len(data objects used for fitting)={self.n_data}"
            )

        X_t = X.copy()
        for transformer in self.get_transformers():
            X_t = transformer.transform(X_t)  # type: ignore

        return X_t

    def fit_transform(
        self,
        X: list[Data] | Data,
        sample_dims: Dims,
        weights: list[Data] | Data | None = None,
    ) -> DataArray:
        # Take advantage of the fact that `.fit()` already transforms the data
        # to avoid duplicate computation
        self, X = self._fit_algorithm(X, sample_dims, weights)
        return X

    def inverse_transform_data(self, X: DataArray) -> list[Data] | Data:
        """Inverse transform the data.

        Parameters:
        -------------
        data: xr.DataArray
            Input data.

        Returns
        -------
        xr.DataArray or xr.Dataset or list of xr.DataArray
            The inverse transformed data.

        """
        X_it = X.copy()
        for transformer in self.get_transformers(inverse=True):
            X_it = transformer.inverse_transform_data(X_it)

        return self._process_output(X_it)

    def inverse_transform_components(self, X: DataArray) -> list[Data] | Data:
        """Inverse transform the components.

        Parameters:
        -------------
        data: xr.DataArray or list of xarray.DataArray
            Input data.

        Returns
        -------
        xr.DataArray
            The inverse transformed components.

        """
        X_it = X.copy()
        for transformer in self.get_transformers(inverse=True):
            X_it = transformer.inverse_transform_components(X_it)

        return self._process_output(X_it)

    def inverse_transform_scores(self, X: DataArray) -> DataArray:
        """Inverse transform the scores.

        This should be used for scores obtained from the fitted data.

        Parameters:
        -------------
        X: xr.DataArray
            Input data.

        Returns
        -------
        xr.DataArray
            The inverse transformed scores.

        """
        X_it = X.copy()
        for transformer in self.get_transformers(inverse=True):
            X_it = transformer.inverse_transform_scores(X_it)

        return X_it

    def inverse_transform_scores_unseen(self, X: DataArray) -> DataArray:
        """Inverse transform the scores.

        This should be used for scores obtained from new data.

        Parameters:
        -------------
        X: xr.DataArray
            Input data.

        Returns
        -------
        xr.DataArray
            The inverse transformed scores.

        """
        X_it = X.copy()
        for transformer in self.get_transformers(inverse=True):
            X_it = transformer.inverse_transform_scores_unseen(X_it)

        return X_it

    def _process_output(self, X: list[Data]) -> list[Data] | Data:
        if self.return_list:
            return X
        else:
            return unwrap_singleton_list(X)

    def _set_return_list(self, X):
        if isinstance(X, (list, tuple)):
            self.return_list = True
        else:
            self.return_list = False

    def serialize(self) -> DataTree:
        """Serialize the necessary attributes of the fitted pre-processor
        and all transformers to a Dataset."""
        # Serialize the preprocessor as the root node
        dt = self._serialize()
        dt.name = "preprocessor"

        # Serialize all transformers
        names = list(self.transformer_types().keys())
        transformers = self.get_transformers()

        for name, transformer_obj in zip(names, transformers):
            dt_transformer = DataTree()
            if isinstance(transformer_obj, GenericListTransformer):
                dt_transformer["transformers"] = DataTree()
                # Loop through list transformer objects and assign a dummy key
                for i, transformer in enumerate(transformer_obj.transformers):
                    dt_transformer.transformers[str(i)] = transformer.serialize()
            else:
                dt_transformer = transformer_obj.serialize()
            # Place the serialized transformer in the tree
            dt[name] = dt_transformer
            dt[name].parent = dt

        return dt

    @classmethod
    def deserialize(cls, dt: DataTree) -> Self:
        """Deserialize from a DataTree representation of the preprocessor
        and all attached Transformers."""
        # Create the parent preprocessor
        preprocessor = cls._deserialize(dt)

        # Loop through all transformers and deserialize
        names = list(preprocessor.transformer_types().keys())
        transformers = preprocessor.get_transformers()

        for name, transformer_obj in zip(names, transformers):
            if isinstance(transformer_obj, GenericListTransformer):
                # Recreate list transformers sequentially
                for transformer in dt[name].transformers.values():
                    deserialized = preprocessor.transformer_types()[name].deserialize(
                        transformer
                    )
                    transformer_obj.transformers.append(deserialized)
            else:
                # Recreate single transformer
                deserialized = preprocessor.transformer_types()[name].deserialize(
                    dt[name]
                )
                setattr(preprocessor, name, deserialized)

        return preprocessor
