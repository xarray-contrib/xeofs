Installation
------------

Dependencies
~~~~~~~~~~~~~~~~~~~~~

The following packages are dependencies of ``xeofs``:

**Core Dependencies (Required)**

* Python (3.10 or higher)
* `numpy <https://www.numpy.org/>`__
* `pandas <https://pandas.pydata.org/>`__
* `xarray <http://xarray.pydata.org/>`__
* `dask <https://dask.org/>`__
* `scikit-learn <https://scikit-learn.org/stable/>`__
* `typing-extensions <https://pypi.org/project/typing-extensions/>`__
* `tqdm <https://tqdm.github.io/>`__

**For Specialized Models (Optional)**

* `numba <https://numba.pydata.org/>`__
* `statsmodels <https://www.statsmodels.org/stable/index.html>`__

**For I/O (Optional)**

* `h5netcdf <https://h5netcdf.org/>`__
* `netCDF4 <https://unidata.github.io/netcdf4-python/netCDF4/index.html>`__
* `zarr <https://zarr.readthedocs.io/en/stable/>`__


Instructions
~~~~~~~~~~~~

The ``xeofs`` package can be installed using either the `conda <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html>`__ 
package manager 

.. code-block:: bash

    conda install -c conda-forge xeofs

or the Python package installer `pip <https://pip.pypa.io/en/stable/getting-started/>`__

.. code-block:: bash

    pip install xeofs

Several optional dependencies are required for certain functionality and are not installed by default:

* ``zarr``, ``h5netcdf``, or ``netcdf4`` are necessary for saving and loading models to disk
* ``statsmodels`` is required for all models that inherit from ``CPCCA`` including ``CCA``, ``MCA`` and ``RDA``
* ``numba`` is required for the ``GWPCA`` model

These extras can be automatically included when installing with pip:

.. code-block:: bash

    pip install xeofs[complete]
    # or using individual groups
    pip install xeofs[io,etc]
