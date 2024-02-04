Key Features
==================


There are numerous advantages of using ``xeofs``, including:

- **Multi-Dimensional**: Designed for ``xarray`` objects, it applies dimensionality reduction to multi-dimensional data while maintaining data labels.
- **Dask-Integrated**: Supports large datasets via ``Dask`` xarray objects
- **Extensive Methods**: Offers various dimensionality reduction techniques
- **Adaptable Output**: Provides output corresponding to the type of input, whether single or list of ``xr.DataArray`` or ``xr.Dataset``
- **Missing Values**: Handles ``NaN`` values within the data
- **Bootstrapping**: Comes with a user-friendly interface for model evaluation using bootstrapping
- **Efficient**: Ensures computational efficiency, particularly with large datasets through randomized SVD
- **Modular**: Allows users to implement and incorporate new dimensionality reduction methods



Labeled & Multi-Dimensional
---------------------------------------------

``xeofs`` is specifically designed for ``xarray`` objects. This design choice enables you 
to apply dimensionality reduction techniques to multi-dimensional data, while still maintaining 
data labels. The only requirement for the user is to specify the *sample* dimensions. By 
default, ``xeofs`` assumes any other dimensions present in the data as *feature* dimensions. 
For instance,

.. code-block:: python

  model.fit(data, dim="time")

will assume that the ``data`` object has a *sample* dimension named ``time``, while all other
dimensions are *feature* dimensions to be expressed as latent variables.

.. note::

    In this context, *sample* refers to the dimension representing the number of observations. 
    In contrast, *feature* dimensions denote the variables. The main goal of 
    dimensionality reduction techniques is to minimize the number of *feature* dimensions
    without altering the *sample* dimensions.

.. note::

    Multiple *sample* dimensions can be specified as long as at least one *feature* dimension remains.

Dask Support
----------------

If you handle large datasets that exceed memory capacity, ``xeofs`` is designed to work with ``dask``-backed
``xarray`` objects from end-to-end. By default, ``xeofs`` computes models eagerly, which in some
cases can lead to better performance. However, it is also possible to build and fit models "lazily", meaning
no computation will be carried out until the user calls ``.compute()``. To enable lazy computation, specify
``compute=False`` when initializing the model.

.. note::

    Importantly, ``xeofs`` never loads the input dataset(s) into memory.

Lazy Evaluation
---------------

There are a few tricks, and features that need to be explicitly disabled for lazy evaluation to work. First
is the ``check_nans`` option, which skips checking for full or isolated ``NaNs`` in the data. In this case,
the user is responsible for ensuring that the data is free of ``NaNs`` by first applying e.g. ``.dropna()``
or ``.fillna()``. Second is that lazy mode is incompatible with assessing the fit of a rotator class during
evaluation, becaue the entire ``dask`` task graph must be built up front. Therefore, a lazy rotator model will
run out to the full ``max_iter`` regardless of the specified ``rtol``. For that reason it is recommended to
reduce the number of iterations.

As an example, the following lazily creates a rotated EOF model for an 80GB dataset in about a second, which can
then be evaluated later using ``.compute()``.

.. code-block:: python

  import dask.array as da
  import numpy as np
  import xarray as xr

  from xeofs.models import EOF, EOFRotator

  data = xr.DataArray(
      da.random.random((10000, 1440, 720), chunks=(100, 100, 100)),
      dims=["time", "lon", "lat"],
      coords={
          "time": xr.date_range("2000-01-01", periods=10000, freq="D"),
          "lon": np.linspace(-180, 180, 1440),
          "lat": np.linspace(-90, 90, 720),
      },
  )
  model = EOF(n_modes=5, check_nans=False, compute=False)
  model.fit(data, dim="time")

  rotator = EOFRotator(compute=False, max_iter=20)
  rotator.fit(model)

  # Later, you can compute the model
  # rotator.compute()

Available Methods
-----------------

As a multifaceted toolbox, ``xeofs`` aims to host a variety of dimensionality reduction 
and related decomposition techniques. 
Refer to the :doc:`api` section to discover the methods currently available.

.. note::

    Please note that ``xeofs`` is in its developmental phase. If there's a specific method 
    you'd like to see included, we encourage you to open an issue on `GitHub`_.

Model Serialization
-------------------

``xeofs`` models offer convenient ``save()`` and ``load()`` methods for serializing
fitted models to a portable format. 

.. code-block:: python

  from xeofs.models import EOF

  model = EOF()
  model.fit(data, dim="time")
  model.save("my_model.zarr")

  # Later, you can load the model
  loaded_model = EOF.load("my_model.zarr")

Input Data Compatibility
------------------------

``xeofs`` is tailored to function harmoniously with `xarray` objects. Currently, it supports: 

- Single instances of ``xr.DataArray`` or ``xr.Dataset``
- Lists comprising ``xr.DataArray`` or ``xr.Dataset``

An intelligent feature of ``xeofs`` is its ability to deliver the appropriate output based on 
the input type. For instance, executing PCA on a singular ``xr.DataArray`` will yield a single 
``xr.DataArray`` for the PC components. Conversely, if a list of ``xr.DataArray`` is inputted, 
``xeofs`` will return a list of ``xr.DataArray`` as PC components.

.. warning::
  
  A mixed list containing both ``xr.DataArray`` and ``xr.Dataset`` objects is not currently supported.

.. note::

    Some methods like PCA/EOF analysis can also handle complex-valued input data. However, due to a limitation_ in the 
    underlying ``dask`` library, complex-valued data is not supported when using ``dask``.

.. _limitation: https://github.com/dask/dask/issues/7639

Handling Missing Values
-----------------------

Conventional SVD algorithms aren't typically configured to manage missing values. To address this, 
``xeofs`` will take of missing values (``NaN``) within your data. There are two primary types of missing values:

1. **Full-dimensional**: ``NaNs`` spanning all samples for a specific feature or vice versa.
2. **Isolated**: Occasional or sporadic ``NaNs`` within the dataset.

Consider a 3D dataset with dimensions (time, lon, lat). A full-dimensional ``NaN`` might represent a 
grid point (lon, lat) exhibiting ``NaNs`` across all time steps. Conversely, an isolated 
``NaN`` might indicate a grid point (lon, lat) displaying ``NaNs`` for only certain time steps.

``xeofs`` is adept at handling full-dimensional ``NaNs``. However, it cannot manage isolated ``NaNs``,
which requires the user to make a decision about how to fill or remove features or samples containing isolated
``NaNs``. ``xeofs`` does provide an optional runtime check which will raise an error if isolated ``NaNs`` are
detected, which is enabled by default.

Model Evaluation
----------------

``xeofs`` is dedicated to providing a user-friendly interface for model evaluations using bootstrapping.
Currently, only bootstrapping for PCA/EOF analysis is supported
(for a practical example, see :doc:`auto_examples/3validation/index`).

Computationally Efficient
----------------------------------

Regardless of whether you're dealing with in-memory or out-of-memory data, ``xeofs`` ensures computational efficiency. 
This is achieved using randomized SVD which tends to be faster for large matrices than a full SVD. For more details
you can refer to the `sklearn documentation on PCA`_.

A comparative analysis demonstrates the performance of ``xeofs`` against ``eofs`` 
on a standard laptop using a 3D dataset with time, longitude, and latitude 
dimensions. Results indicate that ``xeofs`` computes moderate datasets 
(10,000 samples by 100,000 features) in under a minute. While ``eofs`` is 
faster for smaller datasets, ``xeofs`` excels with larger datasets, offering 
significant speed advantages. The dashed line marks data sets with about 3 MiB; 
``xeofs`` outpaces ``eofs`` above this size, whereas ``eofs`` is preferable for smaller data sets.

.. image:: perf/timings_dark.png
   :height: 300px
   :width: 750px
   :alt: Comparison of computational times between xeofs and eofs for data sets of varying sizes
   :align: center


.. note::

    You can find the script to run the performance tests here_.


.. _here: https://github.com/xarray-contrib/xeofs/tree/main/docs/perf

Implement Your Own Model
-------------------------

The ``xeofs`` package has been designed with modularity in mind, allowing you to seamlessly incorporate new methods. 
For instance, if you'd like to introduce a new dimensionality reduction technique named ``MyModel``, 
you can achieve this by inheriting the ``_BaseModel`` class and implementing its ``_fit_algorithm()`` method.

Here's a detailed walkthrough on how to incorporate a new model:

1. Inherit the BaseModel
^^^^^^^^^^^^^^^^^^^^^^^^
    
Your new model should inherit from the `_BaseModel` class. This abstract base class enables 
the transformation of any ND ``xarray`` object into a 2D ``xarray.DataArray`` with dimensions 
(sample, feature) and back. Additionally, it grants access to handy parameters like 
``n_modes``, ``standardize``, and ``use_coslat``.

.. code-block:: python

  from xeofs.models._base_model import _BaseModel
  from xeofs.models.decomposer import Decomposer

  class MyModel(_BaseModel):
      def __init__(self, **kwargs):
          super().__init__(**kwargs)


2. Define the Fit Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
Your chosen method's entire operation should be encapsulated within the 
``_fit_algorithm()``. This function is triggered by ``fit()`` and handles the model fitting. 
By this stage, ``xeofs`` has already processed essential preprocessing steps, ranging from 
centering and weighting to stacking and handling ``NaN`` values.

Here's a basic PCA example to illustrate the process:

.. code-block:: python

  def _fit_algorithm(self, data):
      # NOTE: The `data` here is a 2D xarray.DataArray with dimensions (sample, feature).

      # We'll illustrate with a simplified PCA.
      # The goal is to perform an SVD on the `data` matrix.
      decomposer = Decomposer(n_modes=self.n_modes)
      decomposer.fit(data)

      # Extract the necessary components from the decomposer.
      scores = decomposer.U_
      components = decomposer.V_
      singular_values = decomposer.s_

      # Store the data for later access using the internal DataContainer class.
      self.data.add(name="my_singular_values", data=singular_values)
      self.data.add(name="my_components", data=components)
      self.data.add(name="my_scores", data=scores)

      # (Optional) Attach model parameters as attributes to your data.
      self.data.set_attrs(self.attrs)

3. Access the Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
After fitting your model, results can be retrieved by creating a method for each data 
piece. The internal ``Preprocessor`` class can assist with this task, 
ensuring that the retrieved data retains the correct format.

Depending on their dimensions, data types are categorized into four groups:

1. (sample, feature, ...)
2. (sample, ...)
3. (feature, ...)
4. (...)

The `Preprocessor` class offers methods corresponding to the first three data groups:

- ``inverse_transform_data`` for (sample, feature, ...)
- ``inverse_transform_scores`` for (sample, ...)
- ``inverse_transform_components`` for (feature, ...)

For group (4), data can be accessed directly since there's no need for back transformation.

.. code-block:: python

    def my_singular_values(self):
        return self.data.get("my_singular_values")

    def my_components(self):
        return self.preprocessor.inverse_transform_components(
            self.data.get("my_components")
        )

    def my_scores(self):
        return self.preprocessor.inverse_transform_scores(self.data.get("my_scores"))


4. Optional: Implement Transform and Inverse Transform Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While it's required to implement the ``transform`` and ``inverse_transform`` methods for a complete model, 
we'll merely indicate their absence for this example.

.. code-block:: python

  def _transform_algorithm(self, data):
      raise NotImplementedError("This model does not support transform.")

  def _inverse_transform_algorithm(self, scores):
      raise NotImplementedError("This model does not support inverse transform.")


5. Execute the Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With all parts in place, you can now initialize and use the new model:

.. code-block:: python

    model = MyModel(n_modes=3)
    model.fit(t2m, dim="time")
    model.my_components()


Comparison With Other Packages
==============================

``xeofs`` stands among a suite of Python packages dedicated to dimensionality reduction. 
Its development has been influenced by other notable packages, each boasting unique and robust features. 
For instance, `eofs`_, crafted by Andrew Dawson, is renowned for its compatibility with ``Dask`` and ``xarray``, 
offering an intuitive EOF analysis interface with a 1D sample dimension. `xMCA`_ is another cherished 
tool, presenting an interface for Maximum Covariance Analysis in ``xarray``. In contrast, `pyEOF`_ is 
tailored for Varimax-rotated EOF analysis but is restricted to 2D (``pandas``) input data. While all these 
tools are useful in their specific realms, they possess limitations. ``xeofs`` aspires to present a more general 
toolkit for dimensionality reduction techniques.


.. list-table::
   :header-rows: 1

   * - 
     - **xeofs**
     - **eofs**
     - **pyEOF**
     - **xMCA**
   * - xarray Interface
     - ✅
     - ✅
     - ❌
     - ✅
   * - Dask Support
     - ✅
     - ✅
     - ❌
     - ❌
   * - Multi-Dimensional
     - ✅
     - Only 1D sample dim
     - 2D input only
     - Only 1D sample dim
   * - Missing Values
     - ✅
     - ✅
     - ❌
     - ✅
   * - Support for ``xr.Dataset``
     - ✅
     - ❌
     - ❌
     - ❌
   * - Algorithm\ :sup:`1`\
     - Randomized SVD
     - Full SVD
     - Randomized SVD
     - Full SVD
   * - Extensible Code Structure
     - ✅
     - ❌
     - ❌
     - ❌
   * - **Validation**
     -
     - 
     - 
     -
   * - Bootstrapping
     - ✅
     - ❌
     - ❌
     - ❌

\ :sup:`1`\ **Note on the algorithm:** The computational burden of a full SVD decomposition for an m x n matrix is O(min(mn², m²n)). However, the randomized SVD, which identifies only the initial k singular values, notably curtails this complexity to O(m n log(k)), making the randomized SVD, as utilized by ``xeofs``, more suitable for expansive datasets. For an in-depth exploration, refer to the `sklearn docs on PCA`_.


.. _pyEOF: https://github.com/zhonghua-zheng/pyEOF
.. _xMCA: https://github.com/Yefee/xMCA
.. _eofs: https://github.com/ajdawson/eofs
.. _`sklearn documentation on PCA`: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
.. _`GitHub`: https://github.com/xarray-contrib/xeofs/issues

