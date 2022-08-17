##################
API
##################

.. note:: :code:`xeofs` is separated into three different interfaces providing entry points to work with popular data types of `NumPy` :code:`np.ndarray`, `pandas` :code:`pd.DataFrame` and `xarray` :code:`xr.DataArray`.


:code:`xeofs.models` contains all EOF related techniques with a `NumPy` interface. If you rather prefer to work with :code:`pd.DataFrame` or :code:`xr.DataArray` objects, just use the corresponding models in the :code:`xeofs.pandas` or :code:`xeofs.xarray` module.


************
Numpy Arrays
************
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:


   xeofs.models.EOF
   xeofs.models.Rotator
   xeofs.models.Bootstrapper



*****************
pandas DataFrames
*****************
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:


   xeofs.pandas.EOF
   xeofs.pandas.Rotator
   xeofs.pandas.Bootstrapper


*****************
xarray DataArrays
*****************
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:


   xeofs.xarray.EOF
   xeofs.xarray.Rotator
   xeofs.xarray.Bootstrapper
