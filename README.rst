|badge1| |badge2| |badge3| |badge4| |badge5|

.. |badge1| image:: https://img.shields.io/github/v/tag/nicrie/xeofs?label=Release
    :alt: GitHub tag (latest SemVer)
.. |badge2| image:: https://img.shields.io/github/workflow/status/nicrie/xeofs/CI
   :alt: GitHub Workflow Status (event)
.. |badge3| image:: https://readthedocs.org/projects/xeofs/badge/?version=latest
   :target: https://xeofs.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. |badge4| image:: https://img.shields.io/pypi/dm/xeofs
    :alt: PyPI - Downloads
.. |badge5| image:: https://codecov.io/gh/nicrie/xeofs/branch/main/graph/badge.svg?token=8040ZDH6U7
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

__Goal__
Create a Python package that provides simple access to a variety of different
EOF-related techniques through the popular interfaces of NumPy_, pandas_
and xarray_.

************************
Current features (work in progress)
************************
This package currently supports:

+ EOF analysis
+ Rotated_ EOF analysis (Varimax, Promax)

Further features:

+ Perform EOF analyis over arbitrary multi-dimensions using the ``axis`` or ``dim`` parameter. Check out the examples e.g. for S-mode_ or T-mode_ analysis.
+ General weights, including some shorthands_ for
  + Standardized EOF analysis ``norm=True``
  + Area weighting based on cosine of latitude (``weights='coslat'``)

.. _T-mode: https://xeofs.readthedocs.io/en/latest/auto_examples/1uni/plot_eof-tmode.html#sphx-glr-auto-examples-1uni-plot-eof-tmode-py
.. _S-mode: https://xeofs.readthedocs.io/en/latest/auto_examples/1uni/plot_eof-smode.html#sphx-glr-auto-examples-1uni-plot-eof-smode-py
.. _shorthands: https://xeofs.readthedocs.io/en/latest/auto_examples/1uni/plot_weighted_eof.html#sphx-glr-auto-examples-1uni-plot-weighted-eof-py
.. _Rotated: https://xeofs.readthedocs.io/en/latest/auto_examples/1uni/plot_rotated_eof.html#sphx-glr-auto-examples-1uni-plot-rotated-eof-py
************************
Installation
************************
The package can be installed via

.. code-block:: ini

  pip install xeofs

************************
Documentation
************************
Documentation_ is work in progress.

.. _Documentation: https://xeofs.readthedocs.io/en/latest/

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
