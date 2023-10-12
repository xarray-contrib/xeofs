import warnings
from typing import Optional, Sequence, Hashable, Dict, Any, Self, List
from abc import ABC, abstractmethod
from datetime import datetime

from ..preprocessing.preprocessor import Preprocessor
from ..data_container import DataContainer
from ..utils.data_types import DataObject, DataArray, Dims
from .._version import __version__

# Ignore warnings from numpy casting with additional coordinates
warnings.filterwarnings("ignore", message=r"^invalid value encountered in cast*")


class _BaseModel(ABC):
    """
    Abstract base class for EOF model.

    Parameters
    ----------
    n_modes: int, default=10
        Number of modes to calculate.
    standardize: bool, default=False
        Whether to standardize the input data.
    use_coslat: bool, default=False
        Whether to use cosine of latitude for scaling.
    use_weights: bool, default=False
        Whether to use weights.
    sample_name: str, default="sample"
        Name of the sample dimension.
    feature_name: str, default="feature"
        Name of the feature dimension.
    solver: {"auto", "full", "randomized"}, default="auto"
        Solver to use for the SVD computation.
    solver_kwargs: dict, default={}
        Additional keyword arguments to pass to the solver.

    """

    def __init__(
        self,
        n_modes=10,
        standardize=False,
        use_coslat=False,
        use_weights=False,
        sample_name="sample",
        feature_name="feature",
        solver="auto",
        solver_kwargs={},
    ):
        self.sample_name = sample_name
        self.feature_name = feature_name
        # Define model parameters
        self._params = {
            "n_modes": n_modes,
            "standardize": standardize,
            "use_coslat": use_coslat,
            "use_weights": use_weights,
            "solver": solver,
        }
        self._solver_kwargs = solver_kwargs
        self._preprocessor_kwargs = dict(
            sample_name=sample_name, feature_name=feature_name
        )

        # Define analysis-relevant meta data
        self.attrs = {"model": "BaseModel"}
        self.attrs.update(self._params)
        self.attrs.update(
            {
                "software": "xeofs",
                "version": __version__,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        # Initialize the Preprocessor to scale and stack the data
        self.preprocessor = Preprocessor(
            with_std=standardize,
            with_coslat=use_coslat,
            with_weights=use_weights,
            **self._preprocessor_kwargs
        )
        # Initialize the data container that stores the results
        self.data = DataContainer()

    def fit(
        self,
        data: DataObject,
        dim: Sequence[Hashable] | Hashable,
        weights: Optional[DataObject] = None,
    ) -> Self:
        """
        Fit the model to the input data.

        Parameters
        ----------
        data: DataArray | Dataset | List[DataArray]
            Input data.
        dim: Sequence[Hashable] | Hashable
            Specify the sample dimensions. The remaining dimensions
            will be treated as feature dimensions.
        weights: Optional[DataArray | Dataset | List[DataArray]]
            Weighting factors for the input data.

        """
        # Preprocess the data
        data2D: DataArray = self.preprocessor.fit_transform(data, dim, weights)

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

    def transform(self, data: DataObject) -> DataArray:
        """Project data onto the components.

        Parameters
        ----------
        data: DataObject
            Data to be transformed.

        Returns
        -------
        projections: DataArray
            Projections of the data onto the components.

        """
        data2D = self.preprocessor.transform(data)
        data2D = self._transform_algorithm(data2D)
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
        data: DataObject,
        dim: Sequence[Hashable] | Hashable,
        weights: Optional[DataObject] = None,
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

        Returns
        -------
        projections: DataArray
            Projections of the data onto the components.

        """
        return self.fit(data, dim, weights).transform(data)

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

    def scores(self) -> DataArray:
        """Get the scores."""
        scores = self.data["scores"]
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
