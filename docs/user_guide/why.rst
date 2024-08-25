==========
Why xeofs?
==========

Dimensionality reduction and pattern recognition techniques rest on a fundamental idea: data can be represented as a compilation of observations (or samples) of 
certain variables (or features). This forms a 2D matrix. Principal Component Analysis (PCA) is a key example of this, known in climate science as Empirical Orthogonal 
Functions (EOF) analysis, which often is based on a Singular Value Decomposition (SVD) of a 2D matrix.

When examining Earth observations, two characteristics of the data stand out:

1. **Multi-dimensional Nature:** The data spans multiple dimensions, including spatial coordinates (longitude, latitude, height), time metrics (time, steps, forecast lead times), and other categories (variable types, sensors).
2. **Large Volume:** These datasets are often enormous, frequently exceeding the memory capacity of a single computer.

For handling multi-dimensional data in Python, the xarray_ library is exceptional. It manages large datasets efficiently, especially when used with dask_.

However, applying dimensionality reduction techniques poses a challenge. Typically, this requires preprocessing the data into a 2D matrix format using *xarray*, 
which is then fed into packages like `scikit-learn`_ or *dask*. This process often results in the loss of dimension labels, forcing users to manually track 
dimensions -- a cumbersome, error-prone, and time-consuming task. While manageable for large projects, it becomes a significant burden for smaller tasks.

*xeofs* is designed to tackle these challenges. It provides a collection of widely-used dimensionality reduction techniques, focusing on methods prevalent in climate sciences. 
It complements robust libraries like *scikit-learn* but stands out due to its seamless compatibility with both *xarray* and *dask*, ensuring a streamlined and efficient user experience.

.. _xarray: https://docs.xarray.dev/en/stable/index.html
.. _dask: https://dask.org/
.. _scikit-learn: https://scikit-learn.org/stable/index.html