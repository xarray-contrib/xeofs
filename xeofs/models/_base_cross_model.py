from typing import Tuple, Hashable, Sequence, Dict, Optional, List, Literal
from typing_extensions import Self
from abc import ABC, abstractmethod
from datetime import datetime

import dask
import numpy as np
import xarray as xr
from datatree import DataTree
from dask.diagnostics.progress import ProgressBar

from .eof import EOF
from ..preprocessing.preprocessor import Preprocessor
from ..data_container import DataContainer
from ..utils.data_types import DataObject, DataArray, DataSet
from ..utils.io import insert_placeholders, open_model_tree, write_model_tree
from ..utils.xarray_utils import convert_to_dim_type, data_is_dask
from ..utils.sanity_checks import validate_input_type
from .._version import __version__


class _BaseCrossModel(ABC):
    """
    Abstract base class for cross-decomposition models.

    Parameters:
    -------------
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
    n_pca_modes: int, default=None
        Number of PCA modes to calculate.
    compute: bool, default=True
        Whether to compute elements of the model eagerly, or to defer computation.
        If True, four pieces of the fit will be computed sequentially: 1) the
        preprocessor scaler, 2) optional NaN checks, 3) SVD decomposition, 4) scores
        and components.
    sample_name: str, default="sample"
        Name of the new sample dimension.
    feature_name: str, default="feature"
        Name of the new feature dimension.
    solver: {"auto", "full", "randomized"}, default="auto"
        Solver to use for the SVD computation.
    solver_kwargs: dict, default={}
        Additional keyword arguments to passed to the SVD solver function.

    """

    def __init__(
        self,
        n_modes=10,
        center=True,
        standardize=False,
        use_coslat=False,
        check_nans=True,
        n_pca_modes=None,
        compute=True,
        verbose=False,
        sample_name="sample",
        feature_name="feature",
        solver="auto",
        random_state=None,
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
            "n_pca_modes": n_pca_modes,
            "compute": compute,
            "sample_name": sample_name,
            "feature_name": feature_name,
            "solver": solver,
            "random_state": random_state,
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
        self._preprocessor_kwargs = {
            "sample_name": sample_name,
            "feature_name": feature_name,
            "with_center": center,
            "with_std": standardize,
            "with_coslat": use_coslat,
            "check_nans": check_nans,
            "compute": compute,
        }

        # Define analysis-relevant meta data
        self.attrs = {"model": "BaseCrossModel"}
        self.attrs.update(
            {
                "software": "xeofs",
                "version": __version__,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        self.attrs.update(self._params)

        # Initialize preprocessors to scale and stack left (1) and right (2) data
        self.preprocessor1 = Preprocessor(**self._preprocessor_kwargs)
        self.preprocessor2 = Preprocessor(**self._preprocessor_kwargs)

        # Initialize the data container that stores the results
        self.data = DataContainer()

        # Initialize PCA objects
        self.pca1 = (
            EOF(n_modes=n_pca_modes, compute=self._params["compute"], check_nans=False)
            if n_pca_modes
            else None
        )
        self.pca2 = (
            EOF(n_modes=n_pca_modes, compute=self._params["compute"], check_nans=False)
            if n_pca_modes
            else None
        )

    def get_serialization_attrs(self) -> Dict:
        return dict(
            data=self.data,
            preprocessor1=self.preprocessor1,
            preprocessor2=self.preprocessor2,
        )

    def fit(
        self,
        data1: DataObject,
        data2: DataObject,
        dim: Hashable | Sequence[Hashable],
        weights1: Optional[DataObject] = None,
        weights2: Optional[DataObject] = None,
    ) -> Self:
        """
        Fit the model to the data.

        Parameters
        ----------
        data1: DataArray | Dataset | List[DataArray]
            Left input data.
        data2: DataArray | Dataset | List[DataArray]
            Right input data.
        dim: Hashable | Sequence[Hashable]
            Define the sample dimensions. The remaining dimensions
            will be treated as feature dimensions.
        weights1: Optional[DataObject]
            Weights to be applied to the left input data.
        weights2: Optional[DataObject]
            Weights to be applied to the right input data.

        """
        validate_input_type(data1)
        validate_input_type(data2)
        if weights1 is not None:
            validate_input_type(weights1)
        if weights2 is not None:
            validate_input_type(weights2)

        self.sample_dims = convert_to_dim_type(dim)
        # Preprocess data1
        data1 = self.preprocessor1.fit_transform(data1, self.sample_dims, weights1)
        # Preprocess data2
        data2 = self.preprocessor2.fit_transform(data2, self.sample_dims, weights2)

        self._fit_algorithm(data1, data2)

        if self._params["compute"]:
            self.data.compute()

        return self

    def transform(
        self, data1: Optional[DataObject] = None, data2: Optional[DataObject] = None
    ) -> Sequence[DataArray]:
        """
        Abstract method to transform the data.


        """
        if data1 is None and data2 is None:
            raise ValueError("Either data1 or data2 must be provided.")

        if data1 is not None:
            validate_input_type(data1)
            # Preprocess data1
            data1 = self.preprocessor1.transform(data1)
        if data2 is not None:
            validate_input_type(data2)
            # Preprocess data2
            data2 = self.preprocessor2.transform(data2)

        return self._transform_algorithm(data1, data2)

    @abstractmethod
    def _fit_algorithm(self, data1: DataArray, data2: DataArray) -> Self:
        """
        Fit the model to the preprocessed data. This method needs to be implemented in the respective
        subclass.

        Parameters
        ----------
        data1, data2: DataArray
            Preprocessed input data of two dimensions: (`sample_name`, `feature_name`)

        """
        raise NotImplementedError

    @abstractmethod
    def _transform_algorithm(
        self, data1: Optional[DataArray] = None, data2: Optional[DataArray] = None
    ) -> Sequence[DataArray]:
        """
        Transform the preprocessed data. This method needs to be implemented in the respective
        subclass.

        Parameters
        ----------
        data1, data2: DataArray
            Preprocessed input data of two dimensions: (`sample_name`, `feature_name`)

        """
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(
        self, scores1: DataObject, scores2: DataObject
    ) -> Tuple[DataObject, DataObject]:
        raise NotImplementedError

    def components(self) -> Tuple[DataObject, DataObject]:
        """Get the components."""
        comps1 = self.data["components1"]
        comps2 = self.data["components2"]

        components1: DataObject = self.preprocessor1.inverse_transform_components(
            comps1
        )
        components2: DataObject = self.preprocessor2.inverse_transform_components(
            comps2
        )
        return components1, components2

    def scores(self) -> Tuple[DataArray, DataArray]:
        """Get the scores."""
        scores1 = self.data["scores1"]
        scores2 = self.data["scores2"]

        scores1: DataArray = self.preprocessor1.inverse_transform_scores(scores1)
        scores2: DataArray = self.preprocessor2.inverse_transform_scores(scores2)
        return scores1, scores2

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

    def get_params(self) -> Dict:
        """Get the model parameters."""
        return self._params

    def serialize(self) -> DataTree:
        """Serialize a complete model with its preprocessors."""
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
        model : _BaseCrossModel
            The loaded model.

        """
        dt = open_model_tree(path, engine=engine, **kwargs)
        model = cls.deserialize(dt)
        return model

    def _validate_loaded_data(self, data: DataArray):
        """Optionally check the loaded data for placeholders."""
        pass
