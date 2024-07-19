=============================================
Labeled ND Data
=============================================

xeofs is specifically designed for xarray_ objects. This design choice allows you to apply dimensionality reduction techniques to multi-dimensional data while 
maintaining data labels. The only requirement for the user is to specify the *sample* dimensions. By default, xeofs assumes all other dimensions present in the 
data are *feature* dimensions. For example:

.. code-block:: python
  
  model.fit(data, dim="time")

This will assume that the data object has a *sample* dimension named ``time``, while all other dimensions are *feature* dimensions to be expressed as latent variables.

.. note::

  In this context, *sample* refers to the dimension representing the number of observations. In contrast, *feature* dimensions denote the variables. The main goal of dimensionality reduction techniques is to minimize the number of feature dimensions without altering the sample dimensions.

.. note::

  Multiple *sample* dimensions can be specified as long as at least one *feature* dimension remains.

---------------------------------------------
Input Data Compatibility
---------------------------------------------

xeofs is tailored to function harmoniously with xarray objects. Currently, it supports:

1. Single instances of ``xarray.DataArray`` or ``xarray.Dataset``
2. Lists comprising ``xarray.DataArray`` or ``xarray.Dataset``

An intelligent feature of xeofs is its ability to deliver the appropriate output based on the input type. For instance, executing PCA on a 
single ``xarray.DataArray`` will yield a single ``xarray.DataArray`` for the PC components. Conversely, if a list of ``xarray.DataArray`` is inputted, 
xeofs will return a list of ``xarray.DataArray`` as PC components.

.. warning::
  
  A mixed list containing both xr.DataArray and xr.Dataset objects is not currently supported.

.. warning::

  Some methods like PCA/EOF analysis can also handle complex-valued input data. However, due to a limitation in the underlying dask_ library, complex-valued data is not supported when using dask.



.. _limitation: https://github.com/dask/dask/issues/7639
.. _xarray: https://docs.xarray.dev/en/stable/index.html
.. _dask: https://dask.org/

