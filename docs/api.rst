##################
API
##################

.. note:: :code:`xeofs` is separated into three different interfaces providing entry points to work with popular data types of `NumPy` :code:`np.ndarray`, `pandas` :code:`pd.DataFrame` and `xarray` :code:`xr.DataArray`.


**********************
Numpy | ``np.ndarray``
**********************
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   xeofs.models.EOF
   xeofs.models.Rotator
   xeofs.models.Bootstrapper
   xeofs.models.MCA



*************************
pandas | ``pd.DataFrame``
*************************
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   xeofs.pandas.EOF
   xeofs.pandas.Rotator
   xeofs.pandas.Bootstrapper
   xeofs.pandas.MCA


*************************
xarray | ``xr.DataArray``
*************************
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   xeofs.xarray.EOF
   xeofs.xarray.Rotator
   xeofs.xarray.Bootstrapper
   xeofs.xarray.MCA
