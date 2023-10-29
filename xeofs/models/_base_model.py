import warnings
from typing import Optional, Sequence, Hashable, Dict, Any, List, TypeVar, Tuple
from typing_extensions import Self
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import xarray as xr

from ..preprocessing.preprocessor import Preprocessor
from ..data_container import DataContainer
from ..utils.data_types import DataObject, Data, DataArray, DataSet, DataList, Dims
from ..utils.sanity_checks import validate_input_type
from ..utils.xarray_utils import (
    convert_to_dim_type,
    get_dims,
    feature_ones_like,
    convert_to_list,
    process_parameter,
    _check_parameter_number,
)
from .._version import __version__

# Ignore warnings from numpy casting with additional coordinates
warnings.filterwarnings("ignore", message=r"^invalid value encountered in cast*")

xr.set_options(keep_attrs=True)


class _BaseModel(ABC):
    """
    Abstract base class for EOF model.

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
    sample_name: str, default="sample"
        Name of the sample dimension.
    feature_name: str, default="feature"
        Name of the feature dimension.
    compute: bool, default=True
        Whether to compute the decomposition immediately. This is recommended
        if the SVD result for the first ``n_modes`` can be accommodated in memory, as it
        boosts computational efficiency compared to deferring the computation.
    verbose: bool, default=False
        Whether to show a progress bar when computing the decomposition.
    random_state: Optional[int], default=None
        Seed for the random number generator.
    solver: {"auto", "full", "randomized"}, default="auto"
        Solver to use for the SVD computation.
    solver_kwargs: dict, default={}
        Additional keyword arguments to pass to the solver.

    """

    def __init__(
        self,
        n_modes=10,
        center=True,
        standardize=False,
        use_coslat=False,
        sample_name="sample",
        feature_name="feature",
        compute=True,
        verbose=False,
        random_state=None,
        solver="auto",
        solver_kwargs={},
    ):
        self.n_modes = n_modes
        self.sample_name = sample_name
        self.feature_name = feature_name

        # Define model parameters
        self._params = {
            "n_modes": n_modes,
            "center": center,
            "standardize": standardize,
            "use_coslat": use_coslat,
            "sample_name": sample_name,
            "feature_name": feature_name,
            "random_state": random_state,
            "verbose": verbose,
            "compute": compute,
            "solver": solver,
        }
        self._solver_kwargs = solver_kwargs
        self._solver_kwargs.update(
            {
                "solver": solver,
                "random_state": random_state,
                "compute": compute,
                "verbose": verbose,
            }
        )

        # Define analysis-relevant meta data
        self.attrs = {"model": "BaseModel"}
        self.attrs.update(
            {
                "software": "xeofs",
                "version": __version__,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        self.attrs.update(self._params)

        # Initialize the Preprocessor to scale and stack the data
        self.preprocessor = Preprocessor(
            sample_name=sample_name,
            feature_name=feature_name,
            with_center=center,
            with_std=standardize,
            with_coslat=use_coslat,
        )
        # Initialize the data container that stores the results
        self.data = DataContainer()

    def fit(
        self,
        X: List[Data] | Data,
        dim: Sequence[Hashable] | Hashable,
        weights: Optional[List[Data] | Data] = None,
    ) -> Self:
        """
        Fit the model to the input data.

        Parameters
        ----------
        X: DataArray | Dataset | List[DataArray]
            Input data.
        dim: Sequence[Hashable] | Hashable
            Specify the sample dimensions. The remaining dimensions
            will be treated as feature dimensions.
        weights: Optional[DataArray | Dataset | List[DataArray]]
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

        return self._fit_algorithm(data2D)

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

    def transform(self, data: List[Data] | Data, normalized=True) -> DataArray:
        """Project data onto the components.

        Parameters
        ----------
        data: DataArray | Dataset | List[DataArray]
            Data to be transformed.
        normalized: bool, default=True
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
        return self.preprocessor.inverse_transform_scores(data2D)

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
        data: List[Data] | Data,
        dim: Sequence[Hashable] | Hashable,
        weights: Optional[List[Data] | Data] = None,
        **kwargs
    ) -> DataArray:
        """Fit the model to the input data and project the data onto the components.

        Parameters
        ----------
        data: DataObject
            Input data.
        dim: Sequence[Hashable] | Hashable
            Specify the sample dimensions. The remaining dimensions
            will be treated as feature dimensions.
        weights: Optional[DataObject]
            Weighting factors for the input data.
        **kwargs
            Additional keyword arguments to pass to the transform method.

        Returns
        -------
        projections: DataArray
            Projections of the data onto the components.

        """
        return self.fit(data, dim, weights).transform(data, **kwargs)

    def inverse_transform(self, mode) -> DataObject:
        """Reconstruct the original data from transformed data.

        Parameters
        ----------
        mode: integer, a list of integers, or a slice object.
            The mode(s) used to reconstruct the data. If a scalar is given,
            the data will be reconstructed using the given mode. If a slice
            is given, the data will be reconstructed using the modes in the
            given slice. If a list of integers is given, the data will be reconstructed
            using the modes in the given list.

        Returns
        -------
        data: DataArray | Dataset | List[DataArray]
            Reconstructed data.

        """
        data_reconstructed = self._inverse_transform_algorithm(mode)
        return self.preprocessor.inverse_transform_data(data_reconstructed)

    @abstractmethod
    def _inverse_transform_algorithm(self, mode) -> DataArray:
        """Reconstruct the original data from transformed data.

        Parameters
        ----------
        mode: integer, a list of integers, or a slice object.
            The mode(s) used to reconstruct the data. If a scalar is given,
            the data will be reconstructed using the given mode. If a slice
            is given, the data will be reconstructed using the modes in the
            given slice. If a list of integers is given, the data will be reconstructed
            using the modes in the given list.

        Returns
        -------
        data: DataArray
            Reconstructed 2D data with dimensions (sample_name, feature_name)

        """
        raise NotImplementedError

    def components(self) -> DataObject:
        """Get the components."""
        components = self.data["components"]
        return self.preprocessor.inverse_transform_components(components)

    def scores(self, normalized=True) -> DataArray:
        """Get the scores.

        Parameters
        ----------
        normalized: bool, default=True
            Whether to normalize the scores by the L2 norm.
        """
        scores = self.data["scores"].copy()
        if normalized:
            attrs = scores.attrs.copy()
            scores = scores / self.data["norms"]
            scores.attrs.update(attrs)
            scores.name = "scores"
        return self.preprocessor.inverse_transform_scores(scores)

    def compute(self, verbose: bool = False):
        """Compute and load delayed model results.

        Parameters
        ----------
        verbose : bool
            Whether or not to provide additional information about the computing progress.

        """
        self.data.compute(verbose=verbose)

    def get_params(self) -> Dict[str, Any]:
        """Get the model parameters."""
        return self._params
