from typing import Sequence

import numpy as np
import xarray as xr
from typing_extensions import Self

from ..base_model import BaseModel
from ..data_container import DataContainer
from ..linalg.rotation import promax
from ..preprocessing import Preprocessor, Whitener
from ..utils.data_types import DataArray, DataObject
from ..utils.xarray_utils import argsort_dask, get_deterministic_sign_multiplier
from .cpcca import CPCCA, ComplexCPCCA, HilbertCPCCA


class CPCCARotator(CPCCA):
    """Rotate a solution obtained from ``xe.cross.CPCCA``.

    Rotate the obtained components and scores of a CPCCA model to increase
    interpretability. The algorithm here is based on the approach of Cheng &
    Dunkerton (1995) [1]_ and adapted to the CPCCA framework [2]_.

    Parameters
    ----------
    n_modes : int, default=10
        Specify the number of modes to be rotated.
    power : int, default=1
        Set the power for the Promax rotation. A ``power`` value of 1 results in
        a Varimax rotation.
    max_iter : int or None, default=None
        Determine the maximum number of iterations for the computation of the
        rotation matrix. If not specified, defaults to 1000 if ``compute=True``
        and 100 if ``compute=False``, since we can't terminate a lazy
        computation based using ``rtol``.
    rtol : float, default=1e-8
        Define the relative tolerance required to achieve convergence and
        terminate the iterative process.
    compute : bool, default=True
        Whether to compute the rotation immediately.

    References
    ----------
    .. [1] Cheng, X. & Dunkerton, T. J. Orthogonal Rotation of Spatial Patterns
        Derived from Singular Value Decomposition Analysis. J. Climate 8,
        2631–2643 (1995).
    .. [2] Swenson, E. Continuum Power CCA: A Unified Approach for Isolating
        Coupled Modes. Journal of Climate 28, 1016–1030 (2015).

    Examples
    --------

    Perform a CPCCA analysis:

    >>> model = CPCCA(n_modes=10)
    >>> model.fit(X, Y, dim='time')

    Then, apply varimax rotation to first 5 components and scores:

    >>> rotator = CPCCARotator(n_modes=5)
    >>> rotator.fit(model)

    Retrieve the rotated components and scores:

    >>> rotator.components()
    >>> rotator.scores()

    """

    def __init__(
        self,
        n_modes: int = 10,
        power: int = 1,
        max_iter: int | None = None,
        rtol: float = 1e-8,
        compute: bool = True,
    ):
        BaseModel.__init__(self)

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
        self.attrs.update({"model": "Rotated CPCCA"})
        self.attrs.update(self._params)

        # Attach empty objects
        self.preprocessor1 = Preprocessor()
        self.preprocessor2 = Preprocessor()
        self.whitener1 = Whitener()
        self.whitener2 = Whitener()
        self.data = DataContainer()
        self.model = CPCCA()

        self.sorted = False

    def get_serialization_attrs(self) -> dict:
        return dict(
            data=self.data,
            preprocessor1=self.preprocessor1,
            preprocessor2=self.preprocessor2,
            whitener1=self.whitener1,
            whitener2=self.whitener2,
            model=self.model,
            sorted=self.sorted,
        )

    def _fit_algorithm(self, model) -> Self:
        self.model = model
        self.preprocessor1 = model.preprocessor1
        self.preprocessor2 = model.preprocessor2
        self.whitener1 = model.whitener1
        self.whitener2 = model.whitener2
        self.sample_name = self.model.sample_name
        self.feature_name = self.model.feature_name
        self.sorted = False

        common_feature_dim = "common_feature_dim"
        feature_name = self._get_feature_name()

        n_modes = self._params["n_modes"]
        power = self._params["power"]
        max_iter = self._params["max_iter"]
        rtol = self._params["rtol"]

        # Construct the combined vector of loadings
        # NOTE: In the methodology
        # used by Cheng & Dunkerton (CD95), the combined vectors are "loaded" or
        # weighted with the square root of the singular values, akin to what is
        # done in standard Varimax rotation. While this approach ensures that
        # the resulting projections are still uncorrelated as in the unrotated
        # solution, it does not conserve the squared covariance fraction, i.e.
        # the amount of explained squared covariance can be different before and
        # after rotation. The authors then introduced a so-called "covariance
        # fraction" which is conserved under rotation, but does not have a clear
        # interpretation as the term covariance fraction is only correct when
        # both data sets X and Y are equal and MCA reduces to PCA.
        svalues = self.model.data["singular_values"].sel(mode=slice(1, n_modes))
        scaling = np.sqrt(svalues)

        # Get unrotated singular vectors
        Qx = self.model.data["components1"].sel(mode=slice(1, n_modes))
        Qy = self.model.data["components2"].sel(mode=slice(1, n_modes))

        # Unwhiten and back-transform into physical space
        Qx = self.whitener1.inverse_transform_components(Qx)
        Qy = self.whitener2.inverse_transform_components(Qy)

        # Rename the feature dimension to a common name so that the combined vectors can be concatenated
        Qx = Qx.rename({feature_name[0]: common_feature_dim})
        Qy = Qy.rename({feature_name[1]: common_feature_dim})

        loadings = xr.concat([Qx, Qy], dim=common_feature_dim) * scaling

        # Rotate loadings
        promax_kwargs = {"power": power, "max_iter": max_iter, "rtol": rtol}
        rot_loadings, rot_matrix, phi_matrix = promax(
            loadings=loadings,
            feature_dim=common_feature_dim,
            compute=self._params["compute"],
            **promax_kwargs,
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

        # Rotated (loaded) singular vectors
        Qx_rot = rot_loadings.isel(
            {common_feature_dim: slice(0, Qx.coords[common_feature_dim].size)}
        )
        Qy_rot = rot_loadings.isel(
            {common_feature_dim: slice(Qx.coords[common_feature_dim].size, None)}
        )

        # Rename the common feature dimension to the original feature names
        Qx_rot = Qx_rot.rename({common_feature_dim: feature_name[0]})
        Qy_rot = Qy_rot.rename({common_feature_dim: feature_name[1]})

        # For consistency with the unrotated model classes, we transform the pattern vectors
        # into the whitened PC space
        Qx_rot = self.whitener1.transform_components(Qx_rot)
        Qy_rot = self.whitener2.transform_components(Qy_rot)

        # Normalization factor of singular vectors
        norm1_rot = xr.apply_ufunc(
            np.linalg.norm,
            Qx_rot,
            input_core_dims=[[feature_name[0], "mode"]],
            output_core_dims=[["mode"]],
            exclude_dims={feature_name[0]},
            kwargs={"axis": 0},
            vectorize=False,
            dask="allowed",
        )
        norm2_rot = xr.apply_ufunc(
            np.linalg.norm,
            Qy_rot,
            input_core_dims=[[feature_name[1], "mode"]],
            output_core_dims=[["mode"]],
            exclude_dims={feature_name[1]},
            kwargs={"axis": 0},
            vectorize=False,
            dask="allowed",
        )

        # Rotated (normalized) singular vectors
        Qx_rot = Qx_rot / norm1_rot
        Qy_rot = Qy_rot / norm2_rot

        # CD95 call the quantity "norm1 * norm2" the "explained covariance"
        explained_covariance = norm1_rot * norm2_rot
        squared_covariance = explained_covariance**2

        # Reorder according to squared covariance
        idx_modes_sorted = argsort_dask(squared_covariance, "mode")[::-1]  # type: ignore
        idx_modes_sorted.coords.update(squared_covariance.coords)

        # Rotate scores using rotation matrix
        scores1 = self.model.data["scores1"].sel(mode=slice(1, n_modes))
        scores2 = self.model.data["scores2"].sel(mode=slice(1, n_modes))

        scores1 = self.whitener1.inverse_transform_scores(scores1)
        scores2 = self.whitener2.inverse_transform_scores(scores2)

        # Normalize scores
        scores1 = scores1 / scaling
        scores2 = scores2 / scaling

        RinvT = self._compute_rot_mat_inv_trans(
            rot_matrix, input_dims=("mode_m", "mode_n")
        )
        # Rename dimension mode to ensure that dot product has dimensions (sample x mode) as output
        scores1 = scores1.rename({"mode": "mode_m"})
        scores2 = scores2.rename({"mode": "mode_m"})
        RinvT = RinvT.rename({"mode_n": "mode"})
        scores1_rot = xr.dot(scores1, RinvT, dims="mode_m") * norm1_rot
        scores2_rot = xr.dot(scores2, RinvT, dims="mode_m") * norm2_rot

        # Ensure consitent signs for deterministic output
        modes_sign = get_deterministic_sign_multiplier(rot_loadings, common_feature_dim)
        Qx_rot = Qx_rot * modes_sign
        Qy_rot = Qy_rot * modes_sign
        scores1_rot = scores1_rot * modes_sign
        scores2_rot = scores2_rot * modes_sign

        # Create data container
        self.data.add(
            name="input_data1", data=self.model.data["input_data1"], allow_compute=False
        )
        self.data.add(
            name="input_data2", data=self.model.data["input_data2"], allow_compute=False
        )
        self.data.add(name="components1", data=Qx_rot)
        self.data.add(name="components2", data=Qy_rot)
        self.data.add(name="scores1", data=scores1_rot)
        self.data.add(name="scores2", data=scores2_rot)
        self.data.add(name="squared_covariance", data=squared_covariance)
        self.data.add(
            name="total_squared_covariance",
            data=self.model.data["total_squared_covariance"],
        )

        self.data.add(name="idx_modes_sorted", data=idx_modes_sorted)
        self.data.add(name="norm1", data=norm1_rot)
        self.data.add(name="norm2", data=norm2_rot)
        self.data.add(name="rotation_matrix", data=rot_matrix)
        self.data.add(name="phi_matrix", data=phi_matrix)
        self.data.add(name="modes_sign", data=modes_sign)

        # Assign analysis-relevant meta data
        self.data.set_attrs(self.attrs)

        return self

    def fit(self, model: CPCCA) -> Self:
        """Rotate the solution obtained from ``xe.cross.CPCCA``.

        Parameters
        ----------
        model : ``xe.cross.CPCCA``
            The CPCCA model to be rotated.

        """
        self._fit_algorithm(model)

        if self._params["compute"]:
            self.compute()

        return self

    def transform(
        self,
        X: DataObject | None = None,
        Y: DataObject | None = None,
        normalized: bool = False,
    ) -> DataArray | list[DataArray]:
        """Transform the data.

        Parameters
        ----------
        X, Y: DataObject | None
            Data to be transformed. At least one of them must be provided.
        normalized: bool, default=False
            Whether to return L2 normalized scores.

        Returns
        -------
        Sequence[DataArray] | DataArray
            Transformed data.

        """
        # raise error if no data is provided
        if X is None and Y is None:
            raise ValueError("No data provided. Please provide X and/or Y.")

        n_modes = self._params["n_modes"]
        rot_matrix = self.data["rotation_matrix"]
        RinvT = self._compute_rot_mat_inv_trans(
            rot_matrix, input_dims=("mode_m", "mode_n")
        )
        RinvT = RinvT.rename({"mode_n": "mode"})

        scaling = self.model.data["singular_values"].sel(mode=slice(1, n_modes))
        scaling = np.sqrt(scaling)

        results = []

        if X is not None:
            # Select the (non-rotated) singular vectors of the first dataset
            comps1 = self.model.data["components1"].sel(mode=slice(1, n_modes))

            # Preprocess the data
            comps1 = self.whitener1.inverse_transform_components(comps1)
            X = self.preprocessor1.transform(X)

            # Compute non-rotated scores by projecting the data onto non-rotated components
            projections1 = xr.dot(X, comps1) / scaling
            # Rotate the scores
            projections1 = projections1.rename({"mode": "mode_m"})
            projections1 = xr.dot(projections1, RinvT, dims="mode_m")
            # Reorder according to variance
            if self.sorted:
                projections1 = projections1.isel(
                    mode=self.data["idx_modes_sorted"].values
                ).assign_coords(mode=projections1.mode)
            # Adapt the sign of the scores
            projections1 = projections1 * self.data["modes_sign"]

            # Unscale the scores
            if not normalized:
                projections1 = projections1 * self.data["norm1"]

            # Unstack the projections
            projections1 = self.preprocessor1.inverse_transform_scores(projections1)

            results.append(projections1)

        if Y is not None:
            # Select the (non-rotated) singular vectors of the second dataset
            comps2 = self.model.data["components2"].sel(mode=slice(1, n_modes))

            # Preprocess the data
            comps2 = self.whitener2.inverse_transform_components(comps2)
            Y = self.preprocessor2.transform(Y)

            # Compute non-rotated scores by project the data onto non-rotated components
            projections2 = xr.dot(Y, comps2) / scaling
            # Rotate the scores
            projections2 = projections2.rename({"mode": "mode_m"})
            projections2 = xr.dot(projections2, RinvT, dims="mode_m")
            # Reorder according to variance
            if self.sorted:
                projections2 = projections2.isel(
                    mode=self.data["idx_modes_sorted"].values
                ).assign_coords(mode=projections2.mode)
            # Determine the sign of the scores
            projections2 = projections2 * self.data["modes_sign"]

            # Unscale the scores
            if not normalized:
                projections2 = projections2 * self.data["norm2"]

            # Unstack the projections
            projections2 = self.preprocessor2.inverse_transform_scores(projections2)

            results.append(projections2)

        if len(results) == 0:
            raise ValueError("provide at least one of [`X`, `Y`]")
        elif len(results) == 1:
            return results[0]
        else:
            return results

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

    def _compute_rot_mat_inv_trans(self, rotation_matrix, input_dims) -> xr.DataArray:
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

    def _get_feature_name(self):
        return self.model.feature_name


class ComplexCPCCARotator(CPCCARotator, ComplexCPCCA):
    """Rotate a solution obtained from ``xe.cross.ComplexCPCCA``.

    Rotate the obtained components and scores of a CPCCA model to increase
    interpretability. The algorithm here is based on the approach of Cheng &
    Dunkerton (1995) [1]_ and adapted to the CPCCA framework [2]_.


    Parameters
    ----------
    n_modes : int, default=10
        Specify the number of modes to be rotated.
    power : int, default=1
        Set the power for the Promax rotation. A ``power`` value of 1 results in
        a Varimax rotation.
    max_iter : int, default=1000
        Determine the maximum number of iterations for the computation of the
        rotation matrix.
    rtol : float, default=1e-8
        Define the relative tolerance required to achieve convergence and
        terminate the iterative process.
    squared_loadings : bool, default=False
        Specify the method for constructing the combined vectors of loadings. If
        True, the combined vectors are loaded with the singular values (termed
        "squared loadings"), conserving the squared covariance under rotation.
        This allows estimation of mode importance after rotation. If False, the
        combined vectors are loaded with the square root of the singular values,
        following the method described by Cheng & Dunkerton.
    compute: bool, default=True
        Whether to compute the rotation immediately.

    References
    ----------
    .. [1] Cheng, X. & Dunkerton, T. J. Orthogonal Rotation of Spatial Patterns
        Derived from Singular Value Decomposition Analysis. J. Climate 8,
        2631–2643 (1995).
    .. [3] Swenson, E. Continuum Power CCA: A Unified Approach for Isolating
        Coupled Modes. Journal of Climate 28, 1016–1030 (2015).

    Examples
    --------

    Perform a CPCCA analysis:

    >>> model = ComplexCPCCA(n_modes=10)
    >>> model.fit(X, Y, dim='time')

    Then, apply varimax rotation to first 5 components and scores:

    >>> rotator = ComplexCPCCARotator(n_modes=5)
    >>> rotator.fit(model)

    Retrieve the rotated components and scores:

    >>> rotator.components()
    >>> rotator.scores()

    """

    def __init__(self, **kwargs):
        CPCCARotator.__init__(self, **kwargs)
        self.attrs.update({"model": "Rotated Complex CPCCA"})
        self.model = ComplexCPCCA()


class HilbertCPCCARotator(ComplexCPCCARotator, HilbertCPCCA):
    """Rotate a solution obtained from ``xe.cross.HilbertCPCCA``.

    Rotate the obtained components and scores of a CPCCA model to increase
    interpretability. The algorithm here is based on the approach of Cheng &
    Dunkerton (1995) [1]_ and adapted to the CPCCA framework [2]_.


    Parameters
    ----------
    n_modes : int, default=10
        Specify the number of modes to be rotated.
    power : int, default=1
        Set the power for the Promax rotation. A ``power`` value of 1 results in
        a Varimax rotation.
    max_iter : int, default=1000
        Determine the maximum number of iterations for the computation of the
        rotation matrix.
    rtol : float, default=1e-8
        Define the relative tolerance required to achieve convergence and
        terminate the iterative process.
    squared_loadings : bool, default=False
        Specify the method for constructing the combined vectors of loadings. If
        True, the combined vectors are loaded with the singular values (termed
        "squared loadings"), conserving the squared covariance under rotation.
        This allows estimation of mode importance after rotation. If False, the
        combined vectors are loaded with the square root of the singular values,
        following the method described by Cheng & Dunkerton.
    compute: bool, default=True
        Whether to compute the rotation immediately.

    References
    ----------
    .. [1] Cheng, X. & Dunkerton, T. J. Orthogonal Rotation of Spatial Patterns
        Derived from Singular Value Decomposition Analysis. J. Climate 8,
        2631–2643 (1995).
    .. [3] Swenson, E. Continuum Power CCA: A Unified Approach for Isolating
        Coupled Modes. Journal of Climate 28, 1016–1030 (2015).

    Examples
    --------

    Perform a CPCCA analysis:

    >>> model = HilbertCPCCA(n_modes=10)
    >>> model.fit(X, Y, dim='time')

    Then, apply varimax rotation to first 5 components and scores:

    >>> rotator = HilbertCPCCARotator(n_modes=5)
    >>> rotator.fit(model)

    Retrieve the rotated components and scores:

    >>> rotator.components()
    >>> rotator.scores()

    """

    def __init__(self, **kwargs):
        ComplexCPCCARotator.__init__(self, **kwargs)
        self.attrs.update({"model": "Rotated Hilbert CPCCA"})
        self.model = HilbertCPCCA()

    def transform(
        self, X: DataObject | None = None, Y: DataObject | None = None, normalized=False
    ) -> Sequence[DataArray]:
        """Transform the data."""
        # Here we make use of the Method Resolution Order (MRO) to call the
        # transform method of the first class in the MRO after `CPCCARotator`
        # that has a transform method. In this case it will be `HilbertCPCCA`,
        # which will raise an error because it does not have a transform method.
        return super(CPCCARotator, self).transform(X, Y, normalized)
