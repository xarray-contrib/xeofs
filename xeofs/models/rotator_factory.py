from .eof import EOF, ComplexEOF
from .eof_rotator import ComplexEOFRotator, EOFRotator
from .mca import MCA, ComplexMCA
from .mca_rotator import ComplexMCARotator, MCARotator


class RotatorFactory:
    """Factory class for creating rotators.

    Parameters
    ----------
    n_rot : int
        Number of modes to be rotated.
    power : int
        Defines the power of Promax rotation. Choosing ``power=1`` equals
        a Varimax solution (the default is 1).
    max_iter : int
        Number of maximal iterations for obtaining the rotation matrix
        (the default is 1000).
    rtol : float
        Relative tolerance to be achieved for early stopping the iteration
        process (the default is 1e-8).

    """

    def __init__(self, **kwargs):
        self.params = kwargs
        self._valid_types = (EOF, ComplexEOF, MCA, ComplexMCA)

    def create_rotator(
        self, model: EOF | ComplexEOF | MCA | ComplexMCA
    ) -> EOFRotator | ComplexEOFRotator | MCARotator | ComplexMCARotator:
        """Create a rotator for the given model.

        Parameters
        ----------
        model : xeofs model
            Model to be rotated.

        Returns
        -------
        xeofs Rotator
            Rotator for the given model.
        """
        # We need to check the type of the model instead of isinstance because
        # of inheritance.
        if type(model) is EOF:
            return EOFRotator(**self.params)
        elif type(model) is ComplexEOF:
            return ComplexEOFRotator(**self.params)
        elif type(model) is MCA:
            return MCARotator(**self.params)
        elif type(model) is ComplexMCA:
            return ComplexMCARotator(**self.params)
        else:
            err_msg = f"Invalid model type. Valid types are {self._valid_types}."
            raise TypeError(err_msg)
