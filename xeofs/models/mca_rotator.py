from .cpcca_rotator import ComplexCPCCARotator, CPCCARotator
from .mca import MCA, ComplexMCA


class MCARotator(CPCCARotator, MCA):
    """Rotate a solution obtained from ``xe.models.MCA``.

    Rotated MCA [1]_ is an extension of the standard MCA that applies an additional rotation
    to the computed modes to maximize the variance explained individually by each mode.
    This rotation method enhances interpretability by distributing the explained variance more
    evenly among the modes, making it easier to discern patterns within the data.

    Parameters
    ----------
    n_modes : int, default=10
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
    .. [1] Cheng, X. & Dunkerton, T. J. Orthogonal Rotation of Spatial Patterns Derived from Singular Value Decomposition Analysis. J. Climate 8, 2631–2643 (1995).

    Examples
    --------
    >>> model = MCA(n_modes=5)
    >>> model.fit(da1, da2, dim='time')
    >>> rotator = MCARotator(n_modes=5, power=2)
    >>> rotator.fit(model)
    >>> rotator.components()

    """

    def __init__(
        self,
        n_modes: int = 10,
        power: int = 1,
        max_iter: int | None = None,
        rtol: float = 1e-8,
        compute: bool = True,
    ):
        super().__init__(
            n_modes=n_modes,
            power=power,
            max_iter=max_iter,
            rtol=rtol,
            compute=compute,
        )

        # Define analysis-relevant meta data
        self.attrs.update({"model": "Rotated MCA"})
        self.model = MCA()


class ComplexMCARotator(ComplexCPCCARotator, MCA):
    """Rotate a solution obtained from ``xe.models.ComplexMCA``.

    Rotate the obtained components and scores of a ``ComplexMCA`` model to increase interpretability. The algorithm here is based on the approach of Cheng & Dunkerton (1995) [1]_, Elipot et al. (2017) [2]_ and Rieger et al. (2021) [3]_.

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
    compute: bool, default=True
        Whether to compute the rotation immediately.

    References
    ----------
    .. [1] Cheng, X. & Dunkerton, T. J. Orthogonal Rotation of Spatial Patterns
        Derived from Singular Value Decomposition Analysis. J. Climate 8,
        2631–2643 (1995).
    .. [2] Elipot, S., Frajka-Williams, E., Hughes, C. W., Olhede, S. &
        Lankhorst, M. Observed Basin-Scale Response of the North Atlantic
        Meridional Overturning Circulation to Wind Stress Forcing. Journal of
        Climate 30, 2029–2054 (2017).
    .. [3] Rieger, N., Corral, Á., Olmedo, E. & Turiel, A. Lagged
        Teleconnections of Climate Variables Identified via Complex Rotated
        Maximum Covariance Analysis. Journal of Climate 34, 9861–9878 (2021).



    Examples
    --------
    Perform a Varimax-rotated MCA:
    >>> model = ComplexMCA(n_modes=5)
    >>> model.fit(X, Y, dim='time')
    >>> rotator = ComplexMCARotator(n_modes=5, power=1)
    >>> rotator.fit(model)
    >>> rotator.components()

    """

    def __init__(
        self,
        n_modes: int = 10,
        power: int = 1,
        max_iter: int | None = None,
        rtol: float = 1e-8,
        compute: bool = True,
    ):
        super().__init__(
            n_modes=n_modes,
            power=power,
            max_iter=max_iter,
            rtol=rtol,
            compute=compute,
        )
        self.attrs.update({"model": "Complex Rotated MCA"})
        self.model = ComplexMCA()
