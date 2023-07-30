:sd_hide_title:

==================
Benefits
==================


Key Features
==================

There are numerous advantages of using ``xeofs``, including:

- **Multi-dimensional Analysis**: Perform labeled EOF analysis using the robust functionality of ``xarray``.
- **Scalability**: Efficiently process large datasets with the help of ``dask``.
- **Speed**: Benefit from the quick execution of EOF analysis using ``scipy``'s randomized SVD.
- **Model Validation**: Validate your models through bootstrapping.
- **Modular Code Structure**: Easily incorporate new EOF variants with the package's modular design.


Flexible Data Formats
====================================

``xeofs`` is designed to work seamlessly with a variety of data structures, thereby accommodating a wide range of use cases and applications.
Specifically, it accepts three types of input: 

- ``xr.DataArray``
- ``xr.Dataset``
- a list of ``xr.DataArray``

This flexibility enables you to fully leverage the powerful data structures provided by ``xarray``, making your analyses more streamlined and unconstrained. 


Handling Missing Values
====================================

``xeofs`` provides intelligent handling of missing values (``NaN``) in your data. 

While isolated ``NaNs``, i.e., sporadic missing values within your data grid, 
are not considered valid input, ``xeofs`` is designed to handle full-dimensional ``NaNs`` gracefully. 

A full-dimensional ``NaN`` occurs when an entire grid point 
(for instance, a longitude-latitude point) contains ``NaNs`` across all time steps. A typical scenario would be an ocean cell in land-only data. 
In such instances, ``xeofs`` treats the full-dimensional ``NaNs`` appropriately without disrupting your EOF analysis. 

Comparison With Other Packages
====================================

``xeofs`` joins a collection of Python packages designed for EOF analysis, each with their unique sets of features. `eofs`_, developed by Andrew Dawson, supports fundamental functionalities such as Dask compatibility for handling large datasets and multivariate EOF analysis.

Yet, the demand for EOF analysis extends to a wider range of functionalities, making room for alternative tools. Complex EOF analysis, Maximum Covariance Analysis (MCA), and rotated EOF analysis, for instance, are not covered by `eofs`_. For MCA, `xMCA`_ serves as a reliable option. `pyEOF`_, on the other hand, caters to the need for Varimax-rotated EOF analysis, although it's strictly confined to 2D (pandas) input data, necessitating additional processing. At present, none of the existing packages accommodate complex EOF analysis or any combinations of such methods, such as complex rotated EOF analysis or rotated MCA.

In view of these varied requirements, ``xeofs`` was designed with the intention of offering a comprehensive platform, incorporating the above-mentioned methods, and more, in a single package. Furthermore, ``xeofs`` is designed to be flexible, accommodating different xarray objects as input and allowing easy incorporation of new methods. Notably, it supports fully multidimensional dimensions (both samples and features), providing unique capacity compared to other packages. For example, if your data has dimensions (year, month, lon, lat) and you wish to perform EOF analysis on the time (year, month) dimensions, ``xeofs`` can handle this, enabling EOF analysis in T-mode, not just S-mode.

Additionally, ``xeofs`` offers a simple interface for bootstrapping, useful for model validation. A full comparison of the different packages is provided in the table below.

.. list-table::
   :header-rows: 1

   * - 
     - **xeofs**
     - **eofs**
     - **pyEOF**
     - **xMCA**
   * - **Supported methods**
     -
     - 
     - 
     -
   * - EOF analysis
     - ✅
     - ✅
     - ✅
     - ❌
   * - MCA
     - ✅
     - ❌
     - ❌
     - ✅
   * - Complex
     - ✅
     - ❌
     - ❌
     - ❌
   * - Rotation
     - ✅
     - ❌
     - ✅
     - ❌
   * - Multivariate
     - ✅
     - ✅
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
   * - **Miscellaneous**
     -
     - 
     - 
     -
   * - xarray interface
     - ✅
     - ✅
     - ❌
     - ✅
   * - Multidimensional
     - ✅
     - Only 1D sample dim
     - 2D input only
     - Only 1D sample dim
   * - Dask support
     - ✅
     - ✅
     - ❌
     - ❌
   * - Algorithm\ :sup:`1`\
     - Randomized SVD
     - Full decomposition
     - Randomized or full SVD
     - Full decomposition

\ :sup:`1`\ **Note on the algorithm:** The computational complexity of full SVD decomposition for a m x n matrix is O(min(mn², m²n)). However, randomized SVD, which only finds the first k singular values, significantly reduces this complexity to O(m n log(k)). This makes randomized SVD, as used by ``xeofs``, more efficient for large datasets. For more details, see the `sklearn docs on PCA`_.

.. _pyEOF: https://github.com/zhonghua-zheng/pyEOF
.. _xMCA: https://github.com/Yefee/xMCA
.. _eofs: https://github.com/ajdawson/eofs
.. _`sklearn docs on PCA`: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html


