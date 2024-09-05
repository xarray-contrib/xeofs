=============================================
Dask Support
=============================================

If you handle large datasets that exceed memory capacity, xeofs is designed to work with dask_-backed
xarray_ objects from end-to-end. By default, xeofs computes models eagerly, which in some
cases can lead to better performance. However, it is also possible to build and fit models "lazily", meaning
no computation will be carried out until the user calls ``.compute()``. To enable lazy computation, specify
``compute=False`` when initializing the model.

.. note::

    Importantly, xeofs never loads the input dataset(s) into memory.

---------------------------------------------
Lazy Evaluation
---------------------------------------------

There are a few tricks, and features that need to be explicitly disabled for lazy evaluation to work. First
is the ``check_nans`` option, which skips checking for full or isolated ``NaNs`` in the data. In this case,
the user is responsible for ensuring that the data is free of ``NaNs`` by first applying e.g. ``.dropna()``
or ``.fillna()``. Second is that lazy mode is incompatible with assessing the fit of a rotator class during
evaluation, becaue the entire dask task graph must be built up front. Therefore, a lazy rotator model will
run out to the full ``max_iter`` regardless of the specified ``rtol``. For that reason it is recommended to
reduce the number of iterations.

As an example, the following lazily creates a rotated EOF model for a 10GB dataset in about a second, which can
then be evaluated later using ``.compute()``.

.. warning::
  
  Remember that dask allows you to compute out-of-memory datasets by writing intermediate results to your disk. However, computing a singular value decomposition (SVD) typically requires more memory during computation than the size of the input dataset. In this case, about twice the size of the dataset (~20 GB) was written to disk during the SVD computation. Usually, these results are written to ``/tmp/`` on Linux machines. You can change the default directory by configuring dask, for example, using: ``dask.config.set({"temporary_directory": "/your/temporary/directory"})``

.. code-block:: python

  import dask.array as da
  import numpy as np
  import xarray as xr

  from xeofs.single import EOF, EOFRotator

  data = xr.DataArray(
      da.random.random((5000, 720, 360), chunks=(100, 100, 100)),
      dims=["time", "lon", "lat"],
      coords={
          "time": xr.date_range("2000-01-01", periods=5000, freq="D"),
          "lon": np.linspace(-180, 180, 720),
          "lat": np.linspace(-90, 90, 360),
      },
  )
  model = EOF(n_modes=5, check_nans=False, compute=False)
  model.fit(data, dim="time")

  rotator = EOFRotator(compute=False, max_iter=20)
  rotator.fit(model)

  # Later, you can compute the model
  # rotator.compute()


.. note::

    A standard laptop (Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz) with four cores, each using 3 GB of memory, needs **about 15 minutes** to compute the PCA.


.. _dask: https://dask.org/
.. _xarray: https://docs.xarray.dev/en/stable/index.html