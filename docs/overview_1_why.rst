Why xeofs?
==========

Dimensionality reduction and pattern recognition techniques often rest on a foundational idea: 
data can be represented as a compilation of observations, referred to as *samples*, 
of certain variables, or *features*. This representation effectively forms a 2D matrix. 
A paramount example of this is Principal Component Analysis (PCA). In climate science, PCA is 
more commonly known as Empirical Orthogonal Functions (EOF) analysis, which relies on the 
Singular Value Decomposition (SVD) of a 2D matrix.

When examining Earth observations, we often encounter two distinctive characteristics of the data:

1. **Multi-dimensional Nature** : The data usually spans multiple dimensions, encompassing spatial coordinates (like longitude, latitude, height, etc.), time metrics (such as time, steps, or forecast lead times), and other categorizations (like variable types, sensors, etc.).
2. **Large Volume** : Such datasets are typically vast, frequently surpassing the memory capabilities of a single computer.

For handling multi-dimensional data in Python, the ``xarray`` library stands out. It can manage 
large datasets proficiently, especially when integrated with ``dask``.

However, the challenge arises when applying dimensionality reduction techniques. A conventional 
workflow necessitates preprocessing the data into a 2D matrix format using ``xarray``, which is 
subsequently input into packages like ``scikit-learn`` or ``dask``. This transformation often 
results in the loss of dimension labels. Consequently, users are compelled to manually track 
dimensions—a process that's cumbersome, prone to errors, and time-intensive. While this might 
be manageable for extensive projects, it becomes a burdensome overhead for smaller ventures.

``xeofs`` is engineered to address the challenges stated above. It offers 
a collection of widely-used dimensionality reduction techniques, with a spotlight on methods 
prevalent in climate sciences. In doing so, it complements robust libraries like 
``scikit-learn``. What sets ``xeofs`` apart is its seamless compatibility with 
both ``xarray`` and ``dask``, ensuring a streamlined user experience.