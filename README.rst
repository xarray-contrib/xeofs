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
.. |badge6| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.6323012.svg
   :target: https://doi.org/10.5281/zenodo.6323012
   
=================================
xeofs: EOF analysis and variants
=================================
Empirical orthogonal function (EOF) analysis, more commonly known as
principal component analysis (PCA), is one of the most popular methods
for dimension reduction and structure identification in Earth system sciences.
Due to this popularity, a number of different EOF variants have been developed
over the last few years, either to mitigate some pitfalls of ordinary EOF
analysis (e.g. orthogonality, interpretability, linearity) or to broaden its
scope (e.g. multivariate variants).

Currently, there are several implementations of EOF analysis on GitHub that
facilitate the acceptance and application of this method by the broader
scientific community. Each of these implementations has its own strengths,
which need to be highlighted (please `let me know`_, if I forgot any):

- eofs_: EOF + Multivariate EOF analyis; used to work with ``xarray``; great documentation, clear code structure and solid tests (partly integrated into ``xeofs``)
- pyEOF_: EOF + Rotated EOF analysis with ``xarray``; makes use of fast PCA implementation of  `scikit-learn` (e.g. Randomized PCA) (fully integrated into ``xeofs``)
- xeof_: EOF analysis with support for large data sets using ``dask``
- xMCA_: Maximum Covariance Analysis (MCA), which is a more general form of EOF analysis

.. _eofs: https://github.com/ajdawson/eofs
.. _xeof: https://github.com/dougiesquire/xeof
.. _xMCA: https://github.com/Yefee/xMCA
.. _pyEOF: https://github.com/zzheng93/pyEOF

.. _let me know: niclasrieger@gmail.com


**********
Goal
**********

The goal of ``xeofs`` is to merge these different implementations and to simplify the integration of other existing and future variants of EOF analysis thanks to its modular code structure.
The official name is deliberately chosen to be similar to the other implementations to make it clear that ``xeofs`` is initially nothing revolutionary new in itself. The point is not to distinguish this implementation from the others, but rather to unify (+ extend) already existing implementations.

This project is intended to be a collaborative project of the scientific community and the contribution of EOF variants in the form of pull requests is explicitly encouraged.
If you are interested, just `contact me`_ or open an `Issue`_.

.. _contact me: niclasrieger@gmail.com
.. _Issue: https://github.com/nicrie/xeofs/issues

*************************************
Current features (work in progress)
*************************************
This package currently supports:

+ Interface to NumPy_, pandas_ and xarray_
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
How to use it?
************************
Documentation_ is work in progress. Meanwhile check out some examples_ to get started.

.. _Documentation: https://xeofs.readthedocs.io/en/latest/
.. _examples: https://xeofs.readthedocs.io/en/latest/auto_examples/index.html


************************
Credits
************************

- Project folder structure: yngvem_
- Testing data: xarray_ \& pooch_


.. _NumPy: https://www.numpy.org
.. _pandas: https://pandas.pydata.org
.. _xarray: https://xarray.pydata.org
.. _yngvem: https://github.com/yngvem/python-project-structure
.. _pooch: https://github.com/fatiando/pooch
