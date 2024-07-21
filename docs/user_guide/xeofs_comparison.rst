
=============================================
Comparison With Other Packages
=============================================

xeofs is part of a suite of Python packages dedicated to dimensionality reduction. Its development has been influenced by several notable packages, each with unique and robust features.

For instance:

* eofs_, crafted by Andrew Dawson, is known for its compatibility with Dask and xarray, offering an intuitive EOF analysis interface with a 1D sample dimension.
* xMCA_ provides an interface for Maximum Covariance Analysis in xarray.
* pyEOF_ is tailored for Varimax-rotated EOF analysis but is limited to 2D (pandas) input data.

While all these tools are valuable in their specific realms, they possess certain limitations. xeofs aims to offer a more general toolkit for dimensionality reduction techniques, providing greater flexibility and broader applicability.


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

