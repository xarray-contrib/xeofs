from abc import abstractmethod
from typing import (
    Hashable,
    Sequence,
)

import xarray as xr
from typing_extensions import Self

from ..base_model import BaseModel
from ..data_container import DataContainer
from ..preprocessing.preprocessor import Preprocessor
from ..utils.data_types import DataArray, DataObject
from ..utils.sanity_checks import validate_input_type
from ..utils.xarray_utils import convert_to_dim_type

xr.set_options(keep_attrs=True)


class BaseModelSingleSet(BaseModel):
    """
    Abstract base class for single-set models.

    Parameters
    ----------
    n_modes: int, default=10
        Number of modes to calculate.
    center: bool, default=True
        Whether to center the input data.
    standardize: bool, default=False
        Whether to standardize the input data.
    use_coslat: bool, default=False
        Whether to use cosine of latitude for scaling.
    check_nans : bool, default=True
        If True, remove full-dimensional NaN features from the data, check to ensure
        that NaN features match the original fit data during transform, and check
        for isolated NaNs. Note: this forces eager computation of dask arrays.
        If False, skip all NaN checks. In this case, NaNs should be explicitly removed
        or filled prior to fitting, or SVD will fail.
    sample_name: str, default="sample"
        Name of the sample dimension.
    feature_name: str, default="feature"
        Name of the feature dimension.
    compute : bool, default=True
        Whether to compute elements of the model eagerly, or to defer computation.
        If True, four pieces of the fit will be computed sequentially: 1) the
        preprocessor scaler, 2) optional NaN checks, 3) SVD decomposition, 4) scores
        and components.
    random_state: int | None, default=None
        Seed for the random number generator.
    solver: {"auto", "full", "randomized"}, default="auto"
        Solver to use for the SVD computation.
    solver_kwargs: dict, default={}
        Additional keyword arguments to pass to the SVD solver function.

    """

    def __init__(
        self,
        n_modes=10,
        center=True,
        standardize=False,
        use_coslat=False,
        check_nans=True,
        sample_name="sample",
        feature_name="feature",
        compute=True,
        random_state=None,
        solver="auto",
        solver_kwargs={},
    ):
        super().__init__()

        self.n_modes = n_modes
        self.sample_name = sample_name
        self.feature_name = feature_name

        # Define model parameters
        self._params = {
            "n_modes": n_modes,
            "center": center,
            "standardize": standardize,
            "use_coslat": use_coslat,
            "check_nans": check_nans,
            "sample_name": sample_name,
            "feature_name": feature_name,
            "random_state": random_state,
            "compute": compute,
            "solver": solver,
            "solver_kwargs": solver_kwargs,
        }
        self._decomposer_kwargs = {
            "n_modes": n_modes,
            "solver": solver,
            "random_state": random_state,
            "compute": compute,
            "solver_kwargs": solver_kwargs,
        }

        # Define analysis-relevant meta data
        self.attrs.update({"model": "BaseModelSingleSet"})
        self.attrs.update(self._params)

        # Initialize the Preprocessor to scale and stack the data
        self.preprocessor = Preprocessor(
            sample_name=sample_name,
            feature_name=feature_name,
            with_center=center,
            with_std=standardize,
            with_coslat=use_coslat,
            check_nans=check_nans,
            compute=compute,
        )
        # Initialize the data container that stores the results
        self.data = DataContainer()

    def get_serialization_attrs(self) -> dict:
        return dict(
            data=self.data,
            preprocessor=self.preprocessor,
        )

    def fit(
        self,
        X: DataObject,
        dim: Sequence[Hashable] | Hashable,
        weights: DataObject | None = None,
    ) -> Self:
        """
        Fit the model to the input data.

        Parameters
        ----------
        X: DataObject
            Input data.
        dim: Sequence[Hashable] | Hashable
            Specify the sample dimensions. The remaining dimensions
            will be treated as feature dimensions.
        weights: DataObject | None, default=None
            Weighting factors for the input data.

        """
        # Check for invalid types
        validate_input_type(X)
        if weights is not None:
            validate_input_type(weights)

        self.sample_dims = convert_to_dim_type(dim)

        # Preprocess the data & transform to 2D
        data2D: DataArray = self.preprocessor.fit_transform(
            X, self.sample_dims, weights
        )

        self._fit_algorithm(data2D)

        if self._params["compute"]:
            self.data.compute()
            self._post_compute()

        return self

    @abstractmethod
    def _fit_algorithm(self, data: DataArray) -> Self:
        """Fit the model to the input data assuming a 2D DataArray.

        Parameters
        ----------
        data: DataArray
            Input data with dimensions (sample_name, feature_name)

        Returns
        -------
        self: Self
            The fitted model.

        """
        raise NotImplementedError

    def transform(self, data: DataObject, normalized=False) -> DataArray:
        """Project data onto the components.

        Parameters
        ----------
        data: DataObject
            Data to be transformed.
        normalized: bool, default=False
            Whether to normalize the scores by the L2 norm.

        Returns
        -------
        projections: DataArray
            Projections of the data onto the components.

        """
        validate_input_type(data)

        data2D = self.preprocessor.transform(data)
        data2D = self._transform_algorithm(data2D)
        if normalized:
            data2D = data2D / self.data["norms"]
            data2D.name = "scores"
        return self.preprocessor.inverse_transform_scores_unseen(data2D)

    @abstractmethod
    def _transform_algorithm(self, data: DataArray) -> DataArray:
        """Project data onto the components.

        Parameters
        ----------
        data: DataArray
            Input data with dimensions (sample_name, feature_name)

        Returns
        -------
        projections: DataArray
            Projections of the data onto the components.

        """
        raise NotImplementedError

    def fit_transform(
        self,
        data: DataObject,
        dim: Sequence[Hashable] | Hashable,
        weights: DataObject | None = None,
        **kwargs,
    ) -> DataArray:
        """Fit the model to the input data and project the data onto the components.

        Parameters
        ----------
        data: DataObject
            Input data.
        dim: Sequence[Hashable] | Hashable
            Specify the sample dimensions. The remaining dimensions
            will be treated as feature dimensions.
        weights: DataObject | None, default=None
            Weighting factors for the input data.
        **kwargs
            Additional keyword arguments to pass to the transform method.

        Returns
        -------
        projections: DataArray
            Projections of the data onto the components.

        """
        return self.fit(data, dim, weights).transform(data, **kwargs)

    def inverse_transform(
        self, scores: DataArray, normalized: bool = False
    ) -> DataObject:
        """Reconstruct the original data from transformed data.

        Parameters
        ----------
        scores: DataArray
            Transformed data to be reconstructed. This could be a subset
            of the `scores` data of a fitted model, or unseen data. Must
            have a 'mode' dimension.
        normalized: bool, default=False
            Whether the scores data have been normalized by the L2 norm.

        Returns
        -------
        data: DataObject
            Reconstructed data.

        """
        if normalized:
            norms = self.data["norms"].sel(mode=scores.mode)
            scores = scores * norms

        # Handle scalar mode in xr.dot
        if "mode" not in scores.dims:
            scores = scores.expand_dims("mode")

        data_reconstructed = self._inverse_transform_algorithm(scores)

        # Reconstructing the data using a single mode introduces a
        # redundant "mode" coordinate
        if "mode" in data_reconstructed.coords:
            data_reconstructed = data_reconstructed.drop_vars("mode")

        return self.preprocessor.inverse_transform_data(data_reconstructed)

    @abstractmethod
    def _inverse_transform_algorithm(self, scores: DataArray) -> DataArray:
        """Reconstruct the original data from transformed data.

        Parameters
        ----------
        scores: DataObject
            Transformed data to be reconstructed. This could be a subset
            of the `scores` data of a fitted model, or unseen data. Must
            have a 'mode' dimension.

        Returns
        -------
        data: DataArray
            Reconstructed 2D data with dimensions (sample_name, feature_name)

        """
        raise NotImplementedError

    def components(self, normalized: bool = True) -> DataObject:
        """Get the components.

        Parameters
        ----------
        normalized: bool, default=True
            Whether to normalize the components by the L2 norm.

        """
        components = self.data["components"]
        if not normalized:
            name = components.name
            components = components * self.data["norms"]
            components.name = name
        return self.preprocessor.inverse_transform_components(components)

    def scores(self, normalized: bool = False) -> DataArray:
        """Get the scores.

        Parameters
        ----------
        normalized: bool, default=True
            Whether to normalize the scores by the L2 norm.
        """
        scores = self.data["scores"].copy()
        if normalized:
            name = scores.name
            scores = scores / self.data["norms"]
            scores.name = name
        return self.preprocessor.inverse_transform_scores(scores)
