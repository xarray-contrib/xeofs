##################
Models
##################

.. note:: :code:`xeofs` is separated into three different interfaces providing entry points to work with popular data types of `NumPy` :code:`np.ndarray`, `pandas` :code:`pd.DataFrame` and `xarray` :code:`xr.DataArray`.


:code:`xeofs.models` contains all EOF related techniques with a `NumPy` interface. If you rather prefer to work with :code:`pd.DataFrame` or :code:`xr.DataArray` objects, just use the corresponding models in the :code:`xeofs.pandas` or :code:`xeofs.xarray` module.


.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:


   xeofs.models.eof.EOF



*********
pandas
*********
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:


   xeofs.pandas.eof.EOF


*********
xarray
*********
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:


   xeofs.xarray.eof.EOF
