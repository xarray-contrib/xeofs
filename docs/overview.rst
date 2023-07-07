:sd_hide_title:

=========
Overview_
=========

Why xeofs?
----------

Empirical orthogonal function (EOF) analysis, also known as principal component analysis (PCA), 
and related variants are typically based on the 
decomposition of a 2D matrix. 

However, Earth observation data is often multi-dimensional, 
characterized by spatial (longitude, latitude, height etc.), temporal (time, steps, lead times 
in forecasts etc.) and other (variable, sensor etc.) dimensions. ``xarray`` is an excellent 
tool in Python for handling such multi-dimensional datasets. 

By using ``xarray`` for EOF 
analysis and similar techniques, it's possible to speed up the analysis by automatically 
taking care of dimension labels. While numerous Python packages exist for EOF analysis 
in ``xarray``, none fulfilled all personal needs, leading to the creation of ``xeofs``. 

Benefits
--------

There are numerous advantages of using ``xeofs``, including:

- **Multi-dimensional Analysis**: Perform labeled EOF analysis using the robust functionality of ``xarray``.
- **Scalability**: Efficiently process large datasets with the help of ``dask``.
- **Speed**: Benefit from the quick execution of EOF analysis using ``scipy``'s randomized SVD.
- **Model Validation**: Validate your models through bootstrapping.
- **Modular Code Structure**: Easily incorporate new EOF variants with the package's modular design.


Supported methods
-----------------

The supported methods in ``xeofs`` include:

- EOF analysis
- Complex EOF analysis
- Maximum Covariance Analysis (MCA)
- Complex MCA
- Varimax/Promax-rotated solutions for better interpretability


Flexible data formats
----------------------

``xeofs`` is designed to work seamlessly with a variety of data structures, thereby accommodating a wide range of use cases and applications.
Specifically, it accepts three types of input: 

- ``xr.DataArray``
- ``xr.Dataset``
- a list of ``xr.DataArray``

This flexibility enables you to fully leverage the powerful data structures provided by ``xarray``, making your analyses more streamlined and unconstrained. 


Handling missing values
------------------------

``xeofs`` provides intelligent handling of missing values (``NaN``) in your data. 

While isolated ``NaNs``, i.e., sporadic missing values within your data grid, 
are not considered valid input, ``xeofs`` is designed to handle full-dimensional ``NaNs`` gracefully. 

A full-dimensional ``NaN`` occurs when an entire grid point 
(for instance, a longitude-latitude point) contains ``NaNs`` across all time steps. A typical scenario would be an ocean cell in land-only data. 
In such instances, ``xeofs`` treats the full-dimensional ``NaNs`` appropriately without disrupting your EOF analysis. 