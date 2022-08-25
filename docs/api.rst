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
   xeofs.models.ROCK_PCA
   xeofs.models.MCA
   xeofs.models.Rotator
   xeofs.models.MCA_Rotator
   xeofs.models.Bootstrapper



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
   xeofs.pandas.MCA_Rotator
   xeofs.pandas.ROCK_PCA

*************************
xarray | ``xr.DataArray``
*************************
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   xeofs.xarray.EOF
   xeofs.xarray.ROCK_PCA
   xeofs.xarray.MCA
   xeofs.xarray.Rotator
   xeofs.xarray.MCA_Rotator
   xeofs.xarray.Bootstrapper
