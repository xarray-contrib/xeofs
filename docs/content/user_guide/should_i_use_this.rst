==================
Should I Use This?
==================

**Short answer**: It depends.

You may not need to use xeofs if:

- The method you need is already available in another package.
- Your data is naturally 2D and unlabeled.

For example, `scikit-learn`_ offers a wide variety of well-established models for 2D data, `cca-zoo`_ provides multiple CCA options, and `pyeof`_ supports Varimax-rotated PCA.

For multi-dimensional data in xarray_ and dask_, popular tools like eofs_ (for PCA/EOF analysis) and xMCA_ (for Maximum Covariance Analysis) might already cover your needs. Specifically, eofs by Andrew Dawson offers basic support for xarray and dask, but only for single 1D sample dimensions.

However, consider using xeofs if:

- You need :doc:`efficient computation<core_functionalities/efficient>` in Python using randomized linear algebra (~10x faster than eofs for medium to large datasets).
- Your data :doc:`exceeds memory limits<core_functionalities/dask_support>` and requires dask for processing.
- You prefer to work within the familiar ecosystem of xarray DataArrays and Datasets.
- Your data is naturally :doc:`N-dimensional<core_functionalities/labeled_data>` (e.g., time, longitude, latitude, steps, sensors, variables).
- You want to perform analysis along an N-dimensional sample dimension such as in :doc:`T-mode EOF analysis<auto_examples/1single/plot_eof-tmode>`.
- You need specialized dimensionality reduction methods for climate science (e.g. Hilbert PCA, POP analysis :doc:`and more<../api_reference/index>`).

Below is an overview of some features where xeofs stands out compared to other packages:

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

\ :sup:`1`\ **Note on the algorithm:** The computational burden of a full SVD decomposition for an m x n matrix is O(min(mn², m²n)). owever, the randomized SVD, which identifies only the initial k singular values, notably curtails this complexity to O(m n log(k)), making the randomized SVD, as utilized by xeofs, more suitable for expansive datasets. For an in-depth exploration, refer to the `sklearn docs on PCA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_.


.. _pyEOF: https://github.com/zhonghua-zheng/pyEOF
.. _xMCA: https://github.com/Yefee/xMCA
.. _eofs: https://github.com/ajdawson/eofs
.. _`GitHub`: https://github.com/xarray-contrib/xeofs/issues
.. _xarray: https://docs.xarray.dev/en/stable/index.html
.. _Dask: https://dask.org/
.. _`scikit-learn`: https://scikit-learn.org/
.. _`cca-zoo`: https://cca-zoo.readthedocs.io/en/latest/