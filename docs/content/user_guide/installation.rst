Installation
------------

Required Dependencies
~~~~~~~~~~~~~~~~~~~~~

The following packages are required dependencies:

**Core Dependencies**

* Python (3.10 or higher)
* `numpy <https://www.numpy.org/>`__ 
* `pandas <https://pandas.pydata.org/>`__ 
* `xarray <http://xarray.pydata.org/>`__ 
* `scikit-learn <https://scikit-learn.org/stable/>`__ 
* `statsmodels <https://www.statsmodels.org/stable/index.html>`__ 

**For Performance**

* `dask <https://dask.org/>`__ 
* `numba <https://numba.pydata.org/>`__ 

**For I/O**

* `netCDF4 <https://unidata.github.io/netcdf4-python/netCDF4/index.html>`__ 
* `zarr <https://zarr.readthedocs.io/en/stable/>`__ 
* `xarray-datatree <https://github.com/xarray-contrib/datatree>`__

**Miscellaneous**

* `typing-extensions <https://pypi.org/project/typing-extensions/>`__ 
* `tqdm <https://tqdm.github.io/>`__ 

Instructions
~~~~~~~~~~~~

The ``xeofs`` package can be installed using either the `conda <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html>`__ 
package manager 

.. code-block:: bash

    conda install -c conda-forge xeofs

or the Python package installer `pip <https://pip.pypa.io/en/stable/getting-started/>`__

.. code-block:: bash

    pip install xeofs
