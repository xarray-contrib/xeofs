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

If you handle large datasets that exceed memory capacity, you can pass a ``Dask`` ``xarray`` object to ``xeofs``. 
Contrary to expectations, ``xeofs`` computes the matrix decomposition results, assuming the output can fit 
into memory. This behavior is the default because deferring computations usually causes substantial 
increase in computational time since any derivative of the decomposition will have to recompute the entire SVD decomposition. 

.. code-block:: python

  model = xe.models.EOF()
  model.fit(data, dim="time")  # <- data is a Dask object
  
  model.components() # Returns a non-Dask object

However, if you wish to postpone computation, you can achieve this by setting ``compute=False`` during model initialization:

.. code-block:: python

  model = xe.models.EOF(compute=False)
  model.fit(data, dim="time")
  
  model.components() # Returns a Dask object

  model.compute() # Computes all deferred objects

  model.components() # Now, it's no longer a Dask object

.. note::

    Importantly, ``xeofs`` never computes the input data directly.

Available Methods
-----------------

As a multifaceted toolbox, ``xeofs`` aims to host a variety of dimensionality reduction 
and related decomposition techniques. 
Refer to the :doc:`api` section to discover the methods currently available.

.. note::

    Please note that ``xeofs`` is in its developmental phase. If there's a specific method 
    you'd like to see included, we encourage you to open an issue on `GitHub`_.

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

Handling Missing Values
-----------------------

Conventional SVD algorithms aren't typically configured to manage missing values. To address this, 
``xeofs`` will take of missing values (``NaN``) within your data. There are two primary types of missing values:

1. **Full-dimensional**: ``NaNs`` spanning all samples for a specific feature or vice versa.
2. **Isolated**: Occasional or sporadic ``NaNs`` within the dataset.

Consider a 3D dataset with dimensions (time, lon, lat). A full-dimensional ``NaN`` might represent a 
grid point (lon, lat) exhibiting ``NaNs`` across all time steps. Conversely, an isolated 
``NaN`` might indicate a grid point (lon, lat) displaying ``NaNs`` for only certain time steps.

``xeofs`` is adept at handling full-dimensional ``NaNs``. However, it cannot manage isolated ``NaNs``. In situations where isolated ``NaNs`` are detected, ``xeofs`` will raise an error.

Model Evaluation
----------------

``xeofs`` is dedicated to providing a user-friendly interface for model evaluations using bootstrapping. Currently, only bootstrapping for PCA/EOF analysis is supported 
(for a practical example, see :doc:`auto_examples/3validation/index`).

Computationally Efficient
----------------------------------

Regardless of whether you're dealing with in-memory or out-of-memory data, ``xeofs`` ensures computational efficiency. 
This is achieved using randomized SVD, a swift method for large matrix decomposition. For an in-depth understanding, 
you can refer to the `sklearn documentation on PCA`_.

To illustrate, a comparison between the computational times of ``xeofs`` (randomized SVD) and ``eofs`` (full SVD) 
for a 3D dataset with dimensions (time, lon, lat) with a varying number features. For comparison, the number of samples is kept fixed to 1000.
It reveals that ``xeofs`` generally outperforms for datasets with over ~500 features both with and without ``Dask``. 
For datasets with fewer than ~500 features, ``eofs`` tends to be quicker, probably because the computational overhead of ``xeofs`` is too large for small datasets.

.. image:: img/timings_dark.png
   :height: 400px
   :width: 800px
   :alt: Comparison of computational times between xeofs and eofs for data sets of varying sizes
   :align: center



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

  def _inverse_transform_algorithm(self, data):
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
.. _`GitHub`: https://github.com/nicrie/xeofs/issues
