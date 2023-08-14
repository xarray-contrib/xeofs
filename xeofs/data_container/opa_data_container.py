from typing import Optional

import numpy as np
import xarray as xr
from dask.diagnostics.progress import ProgressBar

from xeofs.utils.data_types import DataArray

from ._base_model_data_container import _BaseModelDataContainer
from ..utils.data_types import DataArray


class OPADataContainer(_BaseModelDataContainer):
    """Container to store the results of an Optimal Persistence Analysis (OPA)."""

    def __init__(self):
        super().__init__()
        self._filter_patterns: Optional[DataArray] = None
        self._decorrelation_time: Optional[DataArray] = None

    def set_data(
        self,
        input_data: DataArray,
        components: DataArray,
        scores: DataArray,
        filter_patterns: DataArray,
        decorrelation_time: DataArray,
    ):
        super().set_data(input_data=input_data, components=components, scores=scores)

        self._verify_dims(decorrelation_time, ("mode",))
        self._decorrelation_time = decorrelation_time
        self._decorrelation_time.name = "decorrelation_time"

        self._verify_dims(filter_patterns, ("feature", "mode"))
        self._filter_patterns = filter_patterns
        self._filter_patterns.name = "filter_patterns"

    @property
    def components(self) -> DataArray:
        comps = super().components
        comps.name = "optimal_persistence_pattern"
        return comps

    @property
    def decorrelation_time(self) -> DataArray:
        """Get the decorrelation time."""
        decorr = super()._sanity_check(self._decorrelation_time)
        decorr.name = "decorrelation_time"
        return decorr

    @property
    def filter_patterns(self) -> DataArray:
        """Get the filter patterns."""
        filter_patterns = super()._sanity_check(self._filter_patterns)
        filter_patterns.name = "filter_patterns"
        return filter_patterns

    def compute(self, verbose=False):
        super().compute(verbose)

        if verbose:
            with ProgressBar():
                self._filter_patterns = self.filter_patterns.compute()
                self._decorrelation_time = self.decorrelation_time.compute()
        else:
            self._filter_patterns = self.filter_patterns.compute()
            self._decorrelation_time = self.decorrelation_time.compute()

    def set_attrs(self, attrs: dict):
        """Set the attributes of the results."""
        super().set_attrs(attrs)

        filter_patterns = self._sanity_check(self._filter_patterns)
        decorrelation_time = self._sanity_check(self._decorrelation_time)

        filter_patterns.attrs.update(attrs)
        decorrelation_time.attrs.update(attrs)
