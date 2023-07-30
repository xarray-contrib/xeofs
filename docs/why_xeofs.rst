Why xeofs?
==================

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