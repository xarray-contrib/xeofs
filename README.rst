
.. image:: https://codecov.io/gh/nicrie/xeofs/branch/main/graph/badge.svg?token=8040ZDH6U7
   :target: https://codecov.io/gh/nicrie/xeofs



=================================
xeofs: EOF analysis and variants
=================================
Empirical orthogonal function (EOF) analysis, commonly referred to as
principal component analysis (PCA), is a popular decomposition
technique in climate science. Over the years, a variety of variants
have emerged but the lack of availability of these different methods
in the form of easy-to-use software seems to unnecessarily hinder the
acceptance and uptake of these EOF variants by the broad climate science
community.

************************
Goal (work in progress)
************************
Create a Python package that provides simple access to a variety of different
EOF-related techniques through the popular interfaces of NumPy_, pandas_
and xarray_.



************************
Credits
************************

- General project structure: yngvem_
- Testing data: xarray_ \& pooch_



.. _NumPy: https://www.numpy.org
.. _pandas: https://pandas.pydata.org
.. _xarray: https://xarray.pydata.org
.. _yngvem: https://github.com/yngvem/python-project-structure
.. _pooch: https://github.com/fatiando/pooch
