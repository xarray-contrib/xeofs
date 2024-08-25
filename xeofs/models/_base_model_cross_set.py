from abc import abstractmethod
from typing import Any, Hashable, Sequence

from numpy.random import Generator
from typing_extensions import Self

from ..data_container import DataContainer
from ..preprocessing.preprocessor import Preprocessor
from ..preprocessing.whitener import Whitener
from ..utils.data_types import DataArray, DataObject, GenericType
from ..utils.sanity_checks import validate_input_type
from ..utils.xarray_utils import convert_to_dim_type
from ._base_model import _BaseModel


class _BaseModelCrossSet(_BaseModel):
    # TODO: double check the docstring here, matches __init__?
    """
    Abstract base class for cross-decomposition models.

    Parameters:
    -------------
    n_modes: int
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
    alpha : float, default=1.0
        Parameter to perform fractional whitening of the data. If 0, the data is completely whitened. If 1, the data is not whitened.
    use_pca : bool, default=False
        If True, perform PCA to reduce the dimensionality of the data.
    n_pca_modes : int | float | str, default=0.999
        If int, specifies the number of modes to retain. If float, specifies the fraction of variance in the (whitened) data that should be explained by the retained modes. If "all", all modes are retained.
    init_rank_reduction : float, default=0.3
        Only relevant when `use_pca=True` and `n_modes` is a float, in which case it denotes the fraction of the initial rank to reduce the data to via PCA as a first guess before truncating the solution to the desired fraction of explained variance. This allows for faster computation of PCA via randomized SVD and avoids the need to compute the full SVD.
    compute: bool, default=True
        Whether to compute elements of the model eagerly, or to defer computation.
        If True, four pieces of the fit will be computed sequentially: 1) the
        preprocessor scaler, 2) optional NaN checks, 3) SVD decomposition, 4) scores
        and components.
    random_state: numpy.random.Generator or int, optional
        Seed for the random number generator.
    sample_name: str, default="sample"
        Name of the new sample dimension.
    feature_name: str, default="feature"
        Name of the new feature dimension.
    solver: {"auto", "full", "randomized"}
        Solver to use for the SVD computation.
    solver_kwargs: dict[str, Any], default={}
        Additional keyword arguments to passed to the SVD solver function.

    """

    def __init__(
        self,
        n_modes: int,
        center: Sequence[bool] | bool = True,
        standardize: Sequence[bool] | bool = False,
        use_coslat: Sequence[bool] | bool = False,
        check_nans: Sequence[bool] | bool = True,
        use_pca: Sequence[bool] | bool = True,
        n_pca_modes: Sequence[float | int | str] | float | int | str = 0.999,
        pca_init_rank_reduction: Sequence[float] | float = 0.3,
        alpha: Sequence[float] | float = 1.0,
        solver: Sequence[str] | str = "auto",
        compute: bool = True,
        verbose: bool = False,
        sample_name: str = "sample",
        feature_name: Sequence[str] | str = "feature",
        random_state: Generator | int | None = None,
        solver_kwargs: dict[str, Any] = {},
    ):
        super().__init__()

        # Process parameters
        center = self._process_parameter("center", center, True)
        standardize = self._process_parameter("standardize", standardize, False)
        use_coslat = self._process_parameter("use_coslat", use_coslat, False)
        check_nans = self._process_parameter("check_nans", check_nans, True)
        use_pca = self._process_parameter("use_pca", use_pca, True)
        n_pca_modes = self._process_parameter("n_pca_modes", n_pca_modes, 0.999)
        pca_init_rank_reduction = self._process_parameter(
            "pca_init_rank_reduction", pca_init_rank_reduction, 0.3
        )

        alpha = self._process_parameter("alpha", alpha, 1.0)
        # Ensure that alpha is a float
        alpha = [float(a) for a in alpha]

        # Use feature1 and feature2 throughout the model to refer to the two datasets
        if isinstance(feature_name, str):
            feature_name = [feature_name + str(i + 1) for i in range(2)]
        self._check_parameter_number("feature_name", feature_name)
        if feature_name[0] == feature_name[1]:
            raise ValueError("feature_name must be different for each dataset")

        # Define model parameters
        self.sample_name = sample_name
        self.feature_name = feature_name

        self._params = {
            "n_modes": n_modes,
            "center": center,
            "standardize": standardize,
            "use_coslat": use_coslat,
            "check_nans": check_nans,
            "use_pca": use_pca,
            "n_pca_modes": n_pca_modes,
            "pca_init_rank_reduction": pca_init_rank_reduction,
            "alpha": alpha,
            "sample_name": sample_name,
            "feature_name": feature_name,
            "random_state": random_state,
            "verbose": verbose,
            "compute": compute,
            "solver": solver,
        }

        self._decomposer_kwargs = {
            "n_modes": n_modes,
            "init_rank_reduction": pca_init_rank_reduction,
            "solver": solver,
            "random_state": random_state,
            "compute": compute,
            "verbose": False,
            "solver_kwargs": solver_kwargs,
        }

        # Define analysis-relevant meta data
        self.attrs.update({"model": "BaseModelCrossSet"})
        self.attrs.update(self.get_params())

        # Initialize preprocessors for dataset X and Y
        self.preprocessor1 = Preprocessor(
            sample_name=sample_name,
            feature_name=feature_name[0],
            with_center=center[0],
            with_std=standardize[0],
            with_coslat=use_coslat[0],
            check_nans=check_nans[0],
            compute=compute,
        )

        self.preprocessor2 = Preprocessor(
            sample_name=sample_name,
            feature_name=feature_name[1],
            with_center=center[1],
            with_std=standardize[1],
            with_coslat=use_coslat[1],
            check_nans=check_nans[1],
            compute=compute,
        )

        self.whitener1 = Whitener(
            alpha=alpha[0],
            use_pca=use_pca[0],
            n_modes=n_pca_modes[0],
            init_rank_reduction=pca_init_rank_reduction[0],
            sample_name=sample_name,
            feature_name=feature_name[0],
        )
        self.whitener2 = Whitener(
            alpha=alpha[1],
            use_pca=use_pca[1],
            n_modes=n_pca_modes[1],
            init_rank_reduction=pca_init_rank_reduction[1],
            sample_name=sample_name,
            feature_name=feature_name[1],
        )

        # Initialize the data container that stores the results
        self.data = DataContainer()

    def get_serialization_attrs(self) -> dict:
        return dict(
            data=self.data,
            preprocessor1=self.preprocessor1,
            preprocessor2=self.preprocessor2,
            whitener1=self.whitener1,
            whitener2=self.whitener2,
        )

    @abstractmethod
    def _fit_algorithm(self, X: DataArray, Y: DataArray) -> Self:
        """
        Fit the model to the preprocessed data. This method needs to be implemented in the respective
        subclass.

        Parameters
        ----------
        X, Y: DataArray
            Preprocessed input data of two dimensions: (`sample_name`, `feature_name`)

        """
        raise NotImplementedError

    @abstractmethod
    def _transform_algorithm(
        self, X: DataArray | None = None, Y: DataArray | None = None, **kwargs
    ) -> dict[str, DataArray]:
        """
        Transform the preprocessed data. This method needs to be implemented in the respective
        subclass.

        Parameters
        ----------
        X, Y: DataArray
            Preprocessed input data of two dimensions: (`sample_name`, `feature_name`)

        """
        raise NotImplementedError

    @abstractmethod
    def _inverse_transform_algorithm(
        self, X: DataArray, Y: DataArray, **kwargs
    ) -> tuple[DataArray, DataArray]:
        """
        Reconstruct the original data from transformed data. This method needs to be implemented in the respective
        subclass.

        Parameters
        ----------
        scores1: DataArray
            Transformed left field data to be reconstructed. This could be
            a subset of the `scores` data of a fitted model, or unseen data.
            Must have a 'mode' dimension.
        scores2: DataArray
            Transformed right field data to be reconstructed. This could be
            a subset of the `scores` data of a fitted model, or unseen data.
            Must have a 'mode' dimension.

        Returns
        -------
        Xrec1: DataArray
            Reconstructed data of left field.
        Xrec2: DataArray
            Reconstructed data of right field.

        """
        raise NotImplementedError

    @abstractmethod
    def _predict_algorithm(self, X: DataArray, **kwargs) -> DataArray:
        """Predict the right field from the left field. This method needs to be implemented in the respective subclass."""
        raise NotImplementedError

    @abstractmethod
    def _get_components(self, **kwargs) -> tuple[DataArray, DataArray]:
        """Get the components."""
        raise NotImplementedError

    @abstractmethod
    def _get_scores(self, **kwargs) -> tuple[DataArray, DataArray]:
        """Get the scores."""
        raise NotImplementedError

    def fit(
        self,
        X: DataObject,
        Y: DataObject,
        dim: Hashable | Sequence[Hashable],
        weights_X: DataObject | None = None,
        weights_Y: DataObject | None = None,
    ) -> Self:
        """
        Fit the model to the data.

        Parameters
        ----------
        X: DataObject
            Left input data.
        Y: DataObject
            Right input data.
        dim: Hashable | Sequence[Hashable]
            Define the sample dimensions. The remaining dimensions
            will be treated as feature dimensions.
        weights_X: DataObject | None, default=None
            Weights to be applied to the left input data.
        weights_Y: DataObject | None, default=None
            Weights to be applied to the right input data.

        """
        validate_input_type(X)
        validate_input_type(Y)
        if weights_X is not None:
            validate_input_type(weights_X)
        if weights_Y is not None:
            validate_input_type(weights_Y)

        self.sample_dims = convert_to_dim_type(dim)
        # Preprocess data
        X = self.preprocessor1.fit_transform(X, self.sample_dims, weights_X)
        Y = self.preprocessor2.fit_transform(Y, self.sample_dims, weights_Y)
        # Whiten data
        X = self.whitener1.fit_transform(X)
        Y = self.whitener2.fit_transform(Y)
        # Augment data
        X, y = self._augment_data(X, Y)
        # Fit the model
        self._fit_algorithm(X, Y)

        if self.get_params()["compute"]:
            self.data.compute()

        return self

    def _augment_data(self, X: DataArray, Y: DataArray) -> tuple[DataArray, DataArray]:
        """Optional method to augment the data before fitting."""
        return X, Y

    def transform(
        self, X: DataObject | None = None, Y: DataObject | None = None, normalized=False
    ) -> Sequence[DataArray]:
        """
        Abstract method to transform the data.


        """
        if X is None and Y is None:
            raise ValueError("Either X or Y must be provided.")

        if X is not None:
            validate_input_type(X)
            # Preprocess X
            X = self.preprocessor1.transform(X)
            X = self.whitener1.transform(X)
        if Y is not None:
            validate_input_type(Y)
            # Preprocess Y
            Y = self.preprocessor2.transform(Y)
            Y = self.whitener2.transform(Y)

        data = self._transform_algorithm(X, Y, normalized=normalized)
        data_list = []
        if X is not None:
            X = self.whitener1.inverse_transform_scores_unseen(data["data1"])
            X = self.preprocessor1.inverse_transform_scores_unseen(X)
            data_list.append(X)
        if Y is not None:
            Y = self.whitener2.inverse_transform_scores_unseen(data["data2"])
            Y = self.preprocessor2.inverse_transform_scores_unseen(Y)
            data_list.append(Y)
        return data_list

    def inverse_transform(
        self, X: DataArray, Y: DataArray
    ) -> tuple[DataObject, DataObject]:
        """Reconstruct the original data from transformed data.

        Parameters
        ----------
        X: DataObject
            Transformed left field data to be reconstructed. This could be
            a subset of the `scores` data of a fitted model, or unseen data.
            Must have a 'mode' dimension.
        Y: DataObject
            Transformed right field data to be reconstructed. This could be
            a subset of the `scores` data of a fitted model, or unseen data.
            Must have a 'mode' dimension.

        Returns
        -------
        Xrec1: DataArray | Dataset | List[DataArray]
            Reconstructed data of left field.
        Xrec2: DataArray | Dataset | List[DataArray]
            Reconstructed data of right field.

        """
        # Handle scalar mode in xr.dot
        if "mode" not in X.dims:
            X = X.expand_dims("mode")
        if "mode" not in Y.dims:
            Y = Y.expand_dims("mode")

        X, Y = self._inverse_transform_algorithm(X, Y)

        # Unstack and rescale the data
        X = self.whitener1.inverse_transform_data(X)
        Y = self.whitener2.inverse_transform_data(Y)
        X = self.preprocessor1.inverse_transform_data(X)
        Y = self.preprocessor2.inverse_transform_data(Y)

        return X, Y

    def predict(self, X: DataObject) -> DataArray:
        """Predict the right field from the left field."""

        validate_input_type(X)

        # Preprocess X
        X = self.preprocessor1.transform(X)

        # Whiten X
        X = self.whitener1.transform(X)

        # Predict Y
        Y = self._predict_algorithm(X)

        # Inverse transform Y
        Y = self.whitener2.inverse_transform_scores_unseen(Y)
        Y = self.preprocessor2.inverse_transform_scores_unseen(Y)

        return Y

    def components(self, normalized=True) -> tuple[DataObject, DataObject]:
        """Get the components."""
        Px, Py = self._get_components(normalized=normalized)

        Px = self.whitener1.inverse_transform_components(Px)
        Py = self.whitener2.inverse_transform_components(Py)

        Px: DataObject = self.preprocessor1.inverse_transform_components(Px)
        Py: DataObject = self.preprocessor2.inverse_transform_components(Py)
        return Px, Py

    def scores(self, normalized=False) -> tuple[DataArray, DataArray]:
        """Get the scores."""
        Rx, Ry = self._get_scores(normalized=normalized)

        Rx = self.whitener1.inverse_transform_scores(Rx)
        Ry = self.whitener2.inverse_transform_scores(Ry)

        Rx: DataArray = self.preprocessor1.inverse_transform_scores(Rx)
        Ry: DataArray = self.preprocessor2.inverse_transform_scores(Ry)
        return Rx, Ry

    @staticmethod
    def _check_parameter_number(
        parameter_name: str, parameter: Sequence[GenericType]
    ) -> None:
        if len(parameter) != 2:
            err_msg = (
                f"Expected 2 items for '{parameter_name}', but got {len(parameter)}."
            )
            raise ValueError(err_msg)

    def _process_parameter(
        self,
        parameter_name: str,
        parameter: Sequence[GenericType] | GenericType | None,
        default: GenericType,
    ) -> Sequence[GenericType]:
        n_datasets = 2
        if parameter is None:
            parameter = [default] * n_datasets
        elif not isinstance(parameter, (list, tuple)):
            parameter = [parameter] * n_datasets
        self._check_parameter_number(parameter_name, parameter)
        return parameter