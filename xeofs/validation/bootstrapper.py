from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict

import numpy as np
import xarray as xr
from tqdm import trange

from ..models import EOF
from ..data_container import DataContainer
from ..utils.data_types import DataArray
from .._version import __version__


class _BaseBootstrapper(ABC):
    """Bootstrap a model to obtain significant modes and confidence intervals."""

    def __init__(self, n_bootstraps=20, seed=None):
        self._params = {
            "n_bootstraps": n_bootstraps,
            "seed": seed,
        }

        # Define analysis-relevant meta data
        self.attrs: Dict[str, Any] = {"model": "BaseBootstrapper"}
        self.attrs.update(self._params)
        self.attrs.update(
            {
                "software": "xeofs",
                "version": __version__,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        # Initialize the DataContainer to store the results
        self.data = DataContainer()

    @abstractmethod
    def fit(self, model):
        """Bootstrap a given model."""

        # VARIABLES ARE DEFINED IN THE SUBCLASS ACCORDING TO THE MODEL
        raise NotImplementedError


class EOFBootstrapper(_BaseBootstrapper, EOF):
    """Bootstrap a model to obtain significant modes and confidence intervals."""

    def __init__(self, n_bootstraps=20, seed=None):
        # Call the constructor of _BaseBootstrapper
        super().__init__(n_bootstraps=n_bootstraps, seed=seed)
        self.attrs.update({"model": "Bootstrapped EOF analysis"})

    def fit(self, model: EOF):
        """Bootstrap a given model."""

        self.model = model
        self.preprocessor = model.preprocessor
        sample_name = model.sample_name
        feature_name = model.feature_name

        input_data = model.data["input_data"]
        n_samples = input_data.sample.size

        model_params = model.get_params()
        n_modes: int = model_params["n_modes"]
        n_bootstraps: int = self._params["n_bootstraps"]

        # Set seed for reproducibility
        rng = np.random.default_rng(self._params["seed"])

        # Bootstrap the model
        bst_expvar = []  # type: ignore
        bst_total_variance = []  # type: ignore
        bst_components = []  # type: ignore
        bst_scores = []  # type: ignore
        for i in trange(n_bootstraps):
            # Sample with replacement
            idx_rnd = rng.choice(n_samples, n_samples, replace=True)
            bst_data = input_data.isel({sample_name: idx_rnd})
            # We need to assign the sample coordinates of the real data
            # otherwise the transform() method will raise an error as it
            # tries to align the sample coordinates
            # with the coordinates of the bootstrapped (permutated) data
            bst_data = bst_data.assign_coords({sample_name: input_data[sample_name]})
            # Perform EOF analysis with the subsampled data
            # No scaling because we use the pre-scaled data from the model
            bst_model = EOF(n_modes=n_modes, standardize=False, use_coslat=False)
            bst_model.fit(bst_data, dim="sample")
            # Save results
            expvar = bst_model.data["explained_variance"]
            totvar = bst_model.data["total_variance"]
            components = bst_model.data["components"]
            scores = bst_model.transform(input_data, normalized=False)
            bst_expvar.append(expvar)
            bst_total_variance.append(totvar)
            bst_components.append(components)
            bst_scores.append(scores)

        # Concatenate the bootstrap results along a new dimension
        bst_expvar: DataArray = xr.concat(bst_expvar, dim="n")
        bst_total_variance: DataArray = xr.concat(bst_total_variance, dim="n")
        bst_components: DataArray = xr.concat(bst_components, dim="n")
        bst_scores: DataArray = xr.concat(bst_scores, dim="n")

        # Assign the bootstrap dimension coordinates
        coords_n = np.arange(1, n_bootstraps + 1)
        bst_expvar = bst_expvar.assign_coords(n=coords_n)
        bst_total_variance = bst_total_variance.assign_coords(n=coords_n)
        bst_components = bst_components.assign_coords(n=coords_n)
        bst_scores = bst_scores.assign_coords(n=coords_n)

        # Fix sign of individual components determined by correlation coefficients
        # for a given mode with all the individual bootstrap members
        # NOTE: we use scores as they have typically a lower dimensionality than components
        model_scores = model.data["scores"]
        corr = (
            (bst_scores * model_scores).mean("sample")
            / bst_scores.std("sample")
            / model_scores.std("sample")
        )
        signs = np.sign(corr)
        bst_components = bst_components * signs
        bst_scores = bst_scores * signs

        self.data.add(
            name="input_data", data=model.data["input_data"], allow_compute=False
        )
        self.data.add(name="components", data=bst_components)
        self.data.add(name="scores", data=bst_scores)
        self.data.add(name="norms", data=model.data["norms"])
        self.data.add(name="explained_variance", data=bst_expvar)
        self.data.add(name="total_variance", data=bst_total_variance)

        # Assign the same attributes as the original model
        self.data.set_attrs(self.attrs)
