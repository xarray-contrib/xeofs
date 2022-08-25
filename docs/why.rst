##################
Why ``xeofs``?
##################

EOF analysis and related variants are typically based on a decomposition of a 2D matrix.
When working with Earth observation data, however, the underlying structure
of the data is often multi-dimensional, e.g. they can be described by spatial (longitude, latitude, height etc.),
temporal (time, steps, lead times in forecasts etc.) and/or other (variable, sensor etc.)
dimensions. ``xarray`` provides a great tool to handle these multi-dimensional
data sets in Python. Performing EOF analysis and similar techniques using ``xarray``
could therefore speed up analysis by automatically taking care of dimension labels.

Although there exist already numerous Python packages for EOF analysis in ``xarray``,
none did satify my personal needs which is why I tried to merge existing implementations and extend their funcionalities.


************************************
What can ``xeofs`` do for me?
************************************
* perform your analysis ``xarray`` (support for ``pandas`` and ``numpy`` is also provided)
* analyze multi-dimensional data sets without having to keep track of all dimensions yourself
* investigate the significance of your results
* *planned, but not implemented yet*: work with large data sets with the help of ``dask``

*******************
Supported methods
*******************
* EOF analysis / PCA
* Maximum Covariance Analysis
* ROCK-PCA
* Varimax/Promax rotation for better interpretability
