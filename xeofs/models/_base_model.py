import warnings
from typing import (
    Optional,
    Sequence,
    Hashable,
    Dict,
    Any,
    List,
    Literal,
    TypeVar,
    Tuple,
)
from typing_extensions import Self
from abc import ABC, abstractmethod
from datetime import datetime

import dask
import numpy as np
import xarray as xr
from datatree import DataTree
from dask.diagnostics.progress import ProgressBar

from ..preprocessing.preprocessor import Preprocessor
from ..data_container import DataContainer
from ..utils.data_types import DataObject, Data, DataArray, DataSet, DataList, Dims
from ..utils.io import insert_placeholders, open_model_tree, write_model_tree
from ..utils.sanity_checks import validate_input_type
from ..utils.xarray_utils import (
    convert_to_dim_type,
    get_dims,
    feature_ones_like,
    convert_to_list,
    process_parameter,
    _check_parameter_number,
    data_is_dask,
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
    verbose: bool, default=False
        Whether to show a progress bar when computing the decomposition.
    random_state: Optional[int], default=None
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
            "check_nans": check_nans,
            "sample_name": sample_name,
            "feature_name": feature_name,
            "random_state": random_state,
            "verbose": verbose,
            "compute": compute,
            "solver": solver,
            "solver_kwargs": solver_kwargs,
        }
        self._decomposer_kwargs = {
            "n_modes": n_modes,
            "solver": solver,
            "random_state": random_state,
            "compute": compute,
            "verbose": verbose,
            "solver_kwargs": solver_kwargs,
        }

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
            check_nans=check_nans,
            compute=compute,
        )
        # Initialize the data container that stores the results
        self.data = DataContainer()

    def get_serialization_attrs(self) -> Dict:
        return dict(
            data=self.data,
            preprocessor=self.preprocessor,
        )

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

        self._fit_algorithm(data2D)

        if self._params["compute"]:
            self.data.compute()

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
        data: List[Data] | Data,
        dim: Sequence[Hashable] | Hashable,
        weights: Optional[List[Data] | Data] = None,
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

    def inverse_transform(
        self, scores: DataObject, normalized: bool = True
    ) -> DataObject:
        """Reconstruct the original data from transformed data.

        Parameters
        ----------
        scores: DataObject
            Transformed data to be reconstructed. This could be a subset
            of the `scores` data of a fitted model, or unseen data. Must
            have a 'mode' dimension.
        normalized: bool, default=True
            Whether the scores data have been normalized by the L2 norm.

        Returns
        -------
        data: DataArray | Dataset | List[DataArray]
            Reconstructed data.

        """
        if normalized:
            scores = scores * self.data["norms"]
        data_reconstructed = self._inverse_transform_algorithm(scores)
        return self.preprocessor.inverse_transform_data(data_reconstructed)

    @abstractmethod
    def _inverse_transform_algorithm(self, scores: DataObject) -> DataArray:
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

    def compute(self, verbose: bool = False, **kwargs):
        """Compute and load delayed model results.

        Parameters
        ----------
        verbose : bool
            Whether or not to provide additional information about the computing progress.
        **kwargs
            Additional keyword arguments to pass to `dask.compute()`.
        """
        # find and compute all dask arrays simultaneously to allow dask to optimize the
        # shared graph and avoid duplicate i/o and computations
        dt = self.serialize()

        data_objs = {
            k: v
            for k, v in dt.to_dict().items()
            if data_is_dask(v) and v.attrs.get("allow_compute", True)
        }

        if verbose:
            with ProgressBar():
                (data_objs,) = dask.compute(data_objs, **kwargs)
        else:
            (data_objs,) = dask.compute(data_objs, **kwargs)

        for k, v in data_objs.items():
            dt[k] = DataTree(v)

        # then rebuild the trained model from the computed results
        self._deserialize_attrs(dt)

        self._post_compute()

    def _post_compute(self):
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get the model parameters."""
        return self._params

    def serialize(self) -> DataTree:
        """Serialize a complete model with its preprocessor."""
        # Create a root node for this object with its params as attrs
        ds_root = xr.Dataset(attrs=dict(params=self.get_params()))
        dt = DataTree(data=ds_root, name=type(self).__name__)

        # Retrieve the tree representation of each attached object, or set basic attrs
        for key, attr in self.get_serialization_attrs().items():
            if hasattr(attr, "serialize"):
                dt[key] = attr.serialize()
                dt.attrs[key] = "_is_tree"
            else:
                dt.attrs[key] = attr

        return dt

    def save(
        self,
        path: str,
        overwrite: bool = False,
        save_data: bool = False,
        engine: Literal["zarr", "netcdf4", "h5netcdf"] = "zarr",
        **kwargs,
    ):
        """Save the model.

        Parameters
        ----------
        path : str
            Path to save the model.
        overwrite: bool, default=False
            Whether or not to overwrite the existing path if it already exists.
            Ignored unless `engine="zarr"`.
        save_data : str
            Whether or not to save the full input data along with the fitted components.
        engine : {"zarr", "netcdf4", "h5netcdf"}, default="zarr"
            Xarray backend engine to use for writing the saved model.
        **kwargs
            Additional keyword arguments to pass to `DataTree.to_netcdf()` or `DataTree.to_zarr()`.

        """
        self.compute()

        dt = self.serialize()

        # Remove any raw data arrays at this stage
        if not save_data:
            dt = insert_placeholders(dt)

        write_model_tree(dt, path, overwrite=overwrite, engine=engine, **kwargs)

    @classmethod
    def deserialize(cls, dt: DataTree) -> Self:
        """Deserialize the model and its preprocessors from a DataTree."""
        # Recreate the model with parameters set by root level attrs
        model = cls(**dt.attrs["params"])
        model._deserialize_attrs(dt)
        return model

    def _deserialize_attrs(self, dt: DataTree):
        """Set the necessary attributes of the model from a DataTree."""
        for key, attr in dt.attrs.items():
            if key == "params":
                continue
            elif attr == "_is_tree":
                deserialized_obj = getattr(self, key).deserialize(dt[key])
            else:
                deserialized_obj = attr
            setattr(self, key, deserialized_obj)

    @classmethod
    def load(
        cls,
        path: str,
        engine: Literal["zarr", "netcdf4", "h5netcdf"] = "zarr",
        **kwargs,
    ) -> Self:
        """Load a saved model.

        Parameters
        ----------
        path : str
            Path to the saved model.
        engine : {"zarr", "netcdf4", "h5netcdf"}, default="zarr"
            Xarray backend engine to use for reading the saved model.
        **kwargs
            Additional keyword arguments to pass to `open_datatree()`.

        Returns
        -------
        model : _BaseModel
            The loaded model.

        """
        dt = open_model_tree(path, engine=engine, **kwargs)
        model = cls.deserialize(dt)
        return model

    def _validate_loaded_data(self, data: DataArray):
        """Optionally check the loaded data for placeholders."""
        pass
