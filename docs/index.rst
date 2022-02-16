.. xeofs documentation master file, created by
   sphinx-quickstart on Fri Feb 11 21:42:02 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

####################################
xeofs: EOF analysis and variants
####################################
Empirical orthogonal function (EOF) analysis, commonly referred to as
principal component analysis (PCA), is a popular decomposition
technique in climate science. Over the years, a variety of variants
have emerged but the lack of availability of these different methods
in the form of easy-to-use software seems to unnecessarily hinder the
acceptance and uptake of these EOF variants by the broad climate science
community.

.. note:: Work in progress.

*********
Goal
*********
Create a Python package that provides simple access to a variety of different
EOF-related techniques through the popular interfaces of NumPy_, pandas_
and xarray_.




.. _NumPy: https://www.numpy.org
.. _pandas: https://pandas.pydata.org
.. _xarray: https://xarray.pydata.org


.. toctree::
   :maxdepth: 3
   :caption: Contents:

******************
Documentation
******************

 .. toctree::
    :maxdepth: 2

    installation
    tutorial
    api

..
.. Indices and tables
.. ==================
..
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
