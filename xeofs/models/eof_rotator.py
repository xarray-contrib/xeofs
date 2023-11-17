from datetime import datetime
from typing import Dict

import numpy as np
import xarray as xr
from typing_extensions import Self

from .eof import EOF, ComplexEOF
from ..data_container import DataContainer
from ..preprocessing.preprocessor import Preprocessor
from ..utils.rotation import promax
from ..utils.data_types import DataArray
from ..utils.xarray_utils import argsort_dask, get_deterministic_sign_multiplier
from .._version import __version__


class EOFRotator(EOF):
    """Rotate a solution obtained from ``xe.models.EOF``.

    Rotated EOF analysis (e.g. [1]_) is a variation of standard EOF analysis, which uses a rotation technique
    (Varimax or Promax) on the extracted modes to maximize the variance explained by
    individual modes. This rotation spreads the explained variance more evenly among
    the modes, making them easier to interpret by minimizing the number of significant
    loadings on each mode.

    Parameters
    ----------
    n_modes : int, default=2
        Specify the number of modes to be rotated.
    power : int, default=1
        Set the power for the Promax rotation. A ``power`` value of 1 results
        in a Varimax rotation.
    max_iter : int or None, default=None
        Determine the maximum number of iterations for the computation of the
        rotation matrix. If not specified, defaults to 1000 if ``compute=True``
        and 100 if ``compute=False``, since we can't terminate a lazy computation
        based using ``rtol``.
    rtol : float, default=1e-8
        Define the relative tolerance required to achieve convergence and
        terminate the iterative process.
    compute : bool, default=True
        Whether to compute the rotation immediately.

    References
    ----------
    .. [1] Richman, M.B., 1986. Rotation of principal components. Journal of Climatology 6, 293–335. https://doi.org/10.1002/joc.3370060305

    Examples
    --------
    >>> model = xe.models.EOF(n_modes=10)
    >>> model.fit(data)
    >>> rotator = xe.models.EOFRotator(n_modes=10)
    >>> rotator.fit(model)
    >>> rotator.components()

    """

    def __init__(
        self,
        n_modes: int = 2,
        power: int = 1,
        max_iter: int | None = None,
        rtol: float = 1e-8,
        compute: bool = True,
    ):
        if max_iter is None:
            max_iter = 1000 if compute else 100

        # Define model parameters
        self._params = {
            "n_modes": n_modes,
            "power": power,
            "max_iter": max_iter,
            "rtol": rtol,
            "compute": compute,
        }

        # Define analysis-relevant meta data
        self.attrs = {"model": "Rotated EOF analysis"}
        self.attrs.update(self._params)
        self.attrs.update(
            {
                "software": "xeofs",
                "version": __version__,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        # Attach empty objects
        self.preprocessor = Preprocessor()
        self.data = DataContainer()
        self.model = EOF()

        self.sorted = False

    def get_serialization_attrs(self) -> Dict:
        return dict(
            data=self.data,
            preprocessor=self.preprocessor,
            model=self.model,
            sorted=self.sorted,
        )

    def fit(self, model) -> Self:
        """Rotate the solution obtained from ``xe.models.EOF``.

        Parameters
        ----------
        model : ``xe.models.EOF``
            The EOF model to be rotated.

        """
        self._fit_algorithm(model)

        if self._params["compute"]:
            self.compute()

        return self

    def _fit_algorithm(self, model) -> Self:
        self.model = model
        self.preprocessor = model.preprocessor
        self.sample_name = model.sample_name
        self.feature_name = model.feature_name
        self.sorted = False

        n_modes = self._params.get("n_modes")
        power = self._params.get("power")
        max_iter = self._params.get("max_iter")
        rtol = self._params.get("rtol")

        # Select modes to rotate
        components = model.data["components"].sel(mode=slice(1, n_modes))
        expvar = model.explained_variance().sel(mode=slice(1, n_modes))

        # Rotate loadings
        loadings = components * np.sqrt(expvar)
        promax_kwargs = {"power": power, "max_iter": max_iter, "rtol": rtol}
        rot_loadings, rot_matrix, phi_matrix = promax(
            loadings,
            feature_dim=self.feature_name,
            compute=self._params["compute"],
            **promax_kwargs
        )

        # Assign coordinates to the rotation/correlation matrices
        rot_matrix = rot_matrix.assign_coords(
            mode_m=np.arange(1, rot_matrix.mode_m.size + 1),
            mode_n=np.arange(1, rot_matrix.mode_n.size + 1),
        )
        phi_matrix = phi_matrix.assign_coords(
            mode_m=np.arange(1, phi_matrix.mode_m.size + 1),
            mode_n=np.arange(1, phi_matrix.mode_n.size + 1),
        )

        # Reorder according to variance
        expvar = (abs(rot_loadings) ** 2).sum(self.feature_name)
        idx_modes_sorted = argsort_dask(expvar, "mode")[::-1]
        idx_modes_sorted.coords.update(expvar.coords)

        # Normalize loadings
        rot_components = rot_loadings / np.sqrt(expvar)

        # Compute "pseudo" norms
        n_samples = model.data["input_data"].coords[self.sample_name].size
        norms = (expvar * (n_samples - 1)) ** 0.5
        norms.name = "singular_values"

        # Get unrotated, normalized scores
        svals = model.data["norms"].sel(mode=slice(1, n_modes))
        scores = model.data["scores"].sel(mode=slice(1, n_modes))
        scores = scores / svals
        # Rotate scores
        RinvT = self._compute_rot_mat_inv_trans(
            rot_matrix, input_dims=("mode_m", "mode_n")
        )
        # Rename dimension mode to ensure that dot product has dimensions (sample x mode) as output
        scores = scores.rename({"mode": "mode_m"})
        RinvT = RinvT.rename({"mode_n": "mode"})
        scores = xr.dot(scores, RinvT, dims="mode_m")

        # Scale scores by "pseudo" norms
        scores = scores * norms

        # Ensure consistent signs for deterministic output
        modes_sign = get_deterministic_sign_multiplier(
            rot_components, self.feature_name
        )
        rot_components = rot_components * modes_sign
        scores = scores * modes_sign

        # Store the results
        self.data.add(model.data["input_data"], "input_data", allow_compute=False)
        self.data.add(rot_components, "components")
        self.data.add(scores, "scores")
        self.data.add(norms, "norms")
        self.data.add(expvar, "explained_variance")
        self.data.add(model.data["total_variance"], "total_variance")
        self.data.add(idx_modes_sorted, "idx_modes_sorted")
        self.data.add(rot_matrix, "rotation_matrix")
        self.data.add(phi_matrix, "phi_matrix")
        self.data.add(modes_sign, "modes_sign")

        # Assign analysis-relevant meta data
        self.data.set_attrs(self.attrs)

        return self

    def _post_compute(self):
        """Leave sorting until after compute because it can't be done lazily."""
        self._sort_by_variance()

    def _sort_by_variance(self):
        """Re-sort the mode dimension of all data variables by variance explained."""
        if not self.sorted:
            for key in self.data.keys():
                if "mode" in self.data[key].dims and key != "idx_modes_sorted":
                    self.data[key] = (
                        self.data[key]
                        .isel(mode=self.data["idx_modes_sorted"].values)
                        .assign_coords(mode=self.data[key].mode)
                    )
        self.sorted = True

    def _transform_algorithm(self, data: DataArray) -> DataArray:
        n_modes = self._params["n_modes"]

        svals = self.model.singular_values().sel(mode=slice(1, self._params["n_modes"]))
        pseudo_norms = self.data["norms"]
        # Select the (non-rotated) singular vectors of the first dataset
        components = self.model.data["components"].sel(mode=slice(1, n_modes))

        # Compute non-rotated scores by projecting the data onto non-rotated components
        projections = xr.dot(data, components) / svals
        projections.name = "scores"

        # Rotate the scores
        R = self.data["rotation_matrix"]
        RinvT = self._compute_rot_mat_inv_trans(R, input_dims=("mode_m", "mode_n"))
        projections = projections.rename({"mode": "mode_m"})
        RinvT = RinvT.rename({"mode_n": "mode"})
        projections = xr.dot(projections, RinvT, dims="mode_m")
        # Reorder according to variance
        # this must be done in one line: i) select modes according to their variance, ii) replace coords with modes from 1 ... n
        if self.sorted:
            projections = projections.isel(
                mode=self.data["idx_modes_sorted"].values
            ).assign_coords(mode=projections.mode)

        # Scale scores by "pseudo" norms
        projections = projections * pseudo_norms

        # Adapt the sign of the scores
        projections = projections * self.data["modes_sign"]

        return projections

    def fit_transform(self, model) -> DataArray:
        raise NotImplementedError(
            "The fit_transform method is not implemented for the EOFRotator class."
        )

    def _compute_rot_mat_inv_trans(self, rotation_matrix, input_dims) -> DataArray:
        """Compute the inverse transpose of the rotation matrix.

        For orthogonal rotations (e.g., Varimax), the inverse transpose is equivalent
        to the rotation matrix itself. For oblique rotations (e.g., Promax), the simplification
        does not hold.

        Returns
        -------
        rotation_matrix : xr.DataArray

        """
        if self._params["power"] > 1:
            # inverse matrix
            rotation_matrix = xr.apply_ufunc(
                np.linalg.inv,
                rotation_matrix,
                input_core_dims=[(input_dims)],
                output_core_dims=[(input_dims[::-1])],
                vectorize=False,
                dask="allowed",
            )
            # transpose matrix
            rotation_matrix = rotation_matrix.conj().transpose(*input_dims)
        return rotation_matrix


class ComplexEOFRotator(EOFRotator, ComplexEOF):
    """Rotate a solution obtained from ``xe.models.ComplexEOF``.

    Complex Rotated EOF analysis [1]_ [2]_ [3]_ extends EOF analysis by incorporating both amplitude and phase information
    using a Hilbert transform prior to performing MCA and subsequent Varimax or Promax rotation.
    This adds a further layer of dimensionality to the analysis, allowing for a more nuanced interpretation
    of complex relationships within the data, particularly useful when analyzing oscillatory data.

    Parameters
    ----------
    n_modes : int, default=2
        Specify the number of modes to be rotated.
    power : int, default=1
        Set the power for the Promax rotation. A ``power`` value of 1 results
        in a Varimax rotation.
    max_iter : int, default=1000
        Determine the maximum number of iterations for the computation of the
        rotation matrix.
    rtol : float, default=1e-8
        Define the relative tolerance required to achieve convergence and
        terminate the iterative process.
    compute: bool, default=True
        Whether to compute the rotation immediately.

    References
    ----------
    .. [1] Horel, J., 1984. Complex Principal Component Analysis: Theory and Examples. J. Climate Appl. Meteor. 23, 1660–1673. https://doi.org/10.1175/1520-0450(1984)023<1660:CPCATA>2.0.CO;2
    .. [2] Richman, M.B., 1986. Rotation of principal components. Journal of Climatology 6, 293–335. https://doi.org/10.1002/joc.3370060305
    .. [3] Bloomfield, P., Davis, J.M., 1994. Orthogonal rotation of complex principal components. International Journal of Climatology 14, 759–775. https://doi.org/10.1002/joc.3370140706

    Examples
    --------
    >>> model = xe.models.ComplexEOF(n_modes=10)
    >>> model.fit(data)
    >>> rotator = xe.models.ComplexEOFRotator(n_modes=10)
    >>> rotator.fit(model)
    >>> rotator.components()


    """

    def __init__(
        self,
        n_modes: int = 2,
        power: int = 1,
        max_iter: int = 1000,
        rtol: float = 1e-8,
        compute: bool = True,
    ):
        super().__init__(
            n_modes=n_modes, power=power, max_iter=max_iter, rtol=rtol, compute=compute
        )
        self.attrs.update({"model": "Rotated Complex EOF analysis"})
        self.model = ComplexEOF()

    def _transform_algorithm(self, data: DataArray) -> DataArray:
        # Here we leverage the Method Resolution Order (MRO) to invoke the
        # transform method of the first class in the MRO after EOFRotator that
        # has a transform method. In this case, it will be ComplexEOF. However,
        # please note that `transform` is not implemented for ComplexEOF, so this
        # line of code will actually raise an error.
        return super(EOFRotator, self)._transform_algorithm(data)
