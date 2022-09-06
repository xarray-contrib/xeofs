.. image:: examples/1eof/rotated_eof.jpg
  :align: center
  :width: 800
  :alt: Comparison of standard, Varimax-rotated and Proxmax-rotated EOF analysis for temperature field over North America.

Example_ showing sea surface temperature decomposed via EOF analysis, Varimax rotation and Promax rotation.

.. _Example: https://xeofs.readthedocs.io/en/stable/auto_examples/1eof/plot_rotated_eof.html#sphx-glr-auto-examples-1eof-plot-rotated-eof-py

==================================================
xeofs: Multi-dimensional EOF analysis and variants
==================================================

|badge_build_status| |badge_docs_status| |badge_version_pypi| |badge_conda_version| |badge_downloads| |badge_coverage| |badge_license| |badge_zenodo|

.. |badge_version_pypi| image:: https://img.shields.io/pypi/v/xeofs
   :alt: PyPI
.. |badge_build_status| image:: https://img.shields.io/github/workflow/status/nicrie/xeofs/CI
   :alt: GitHub Workflow Status (event)
.. |badge_docs_status| image:: https://readthedocs.org/projects/xeofs/badge/?version=latest
   :target: https://xeofs.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. |badge_downloads_pypi| image:: https://img.shields.io/pypi/dm/xeofs
    :alt: PyPI - Downloads
.. |badge_coverage| image:: https://codecov.io/gh/nicrie/xeofs/branch/main/graph/badge.svg?token=8040ZDH6U7
    :target: https://codecov.io/gh/nicrie/xeofs
.. |badge_zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.6323012.svg
   :target: https://doi.org/10.5281/zenodo.6323012
   :alt: DOI - Zenodo
.. |badge_license| image:: https://img.shields.io/pypi/l/xeofs
  :alt: License
.. |badge_conda_version| image:: https://img.shields.io/conda/vn/conda-forge/xeofs
   :alt: Conda (channel only)
.. |badge_downloads_conda| image:: https://img.shields.io/conda/dn/conda-forge/xeofs
   :alt: Conda downloads
.. |badge_downloads| image:: https://static.pepy.tech/personalized-badge/xeofs?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads
   :target: https://pepy.tech/project/xeofs
   :alt: Total downloads

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


Available Models
----------------

=====================  ==========  ==========  ==========  ==========  ==========  ==========
Package                 **xeofs**   eofs_       pyEOF_      xeof_       xMCA_       xmca2_
=====================  ==========  ==========  ==========  ==========  ==========  ==========
EOF analysis           ✅           ✅           ✅           ✅           ✅            ✅
Rotated EOF analysis   ✅           ❌           ✅           ❌           ❌            ✅
Complex EOF analysis   ❌           ❌           ❌           ❌           ❌            ✅
`ROCK-PCA`_            ✅           ❌           ❌           ❌           ❌            ❌
Multivariate EOF       ✅           ✅           ❌           ❌           ❌            ❌
MCA                    ✅           ❌           ❌           ❌           ✅            ✅
Rotated MCA            ✅           ❌           ❌           ❌           ❌            ✅
Complex MCA            ❌           ❌           ❌           ❌           ❌            ✅
Multivariate MCA       ✅           ❌           ❌           ❌           ❌            ❌
=====================  ==========  ==========  ==========  ==========  ==========  ==========

.. _ROCK-PCA: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8989964&casa_token=3zKG0dtp-ewAAAAA:FM1CrVISSSqhWEAwPGpQqCgDYccfLG4N-67xNNDzUBQmMvtIOHuC7T6X-TVQgbDg3aDOpKBksg&tag=1


Additional features
----------------------

=====================  ==========  ==========  ==========  ==========  ==========  ==========
Package                 **xeofs**  eofs_       pyEOF_      xeof_       xMCA_       xmca2_
=====================  ==========  ==========  ==========  ==========  ==========  ==========
``numpy`` interface    ✅           ✅           ❌           ❌           ❌           ✅
``pandas`` interface   ✅           ❌           ❌           ❌           ❌           ❌
``xarray`` interface   ✅           ✅           ✅           ✅           ✅           ✅
Fast algorithm         ✅           ❌           ✅           ❌           ❌           ❌
Dask support           ❌           ✅           ❌           ✅           ❌           ❌
Multi-dimensional      ✅           ❌           ❌           ❌           ❌           ❌
Significance analysis  ✅           ❌           ❌           ❌           ❌           ❌
=====================  ==========  ==========  ==========  ==========  ==========  ==========


.. _eofs: https://github.com/ajdawson/eofs
.. _xeof: https://github.com/dougiesquire/xeof
.. _xMCA: https://github.com/Yefee/xMCA
.. _pyEOF: https://github.com/zzheng93/pyEOF
.. _xmca2: https://github.com/nicrie/xmca

.. _let me know: niclasrieger@gmail.com


Why ``xeofs``?
----------------------

The goal of ``xeofs`` is to merge these different implementations and to simplify the integration of other existing and future variants of EOF analysis thanks to its modular code structure.
The official name is deliberately chosen to be similar to the other implementations to make it clear that ``xeofs`` is nothing revolutionary new in itself. The point is not to distinguish this implementation from the others, but rather to unify (+ extend) already existing implementations.

This project is intended to be a collaborative project of the scientific community and the contribution of EOF variants in the form of pull requests is explicitly encouraged.
If you are interested, just `contact me`_ or open an `Issue`_.

.. _contact me: niclasrieger@gmail.com
.. _Issue: https://github.com/nicrie/xeofs/issues



Installation
----------------------

If you are using ``conda``, it is recommend to install via:

.. code-block:: ini

  conda install -c conda-forge xeofs

Alternatively, you can install the package through ``pip``:

.. code-block:: ini

  pip install xeofs


How to use it?
----------------------
Documentation_ is work in progress. Meanwhile check out some examples_ to get started:

+ EOF analysis (S-mode_)
+ EOF analysis (T-mode_)
+ Rotated_ EOF analysis (Varimax, Promax)
+ Weighted_ EOF analysis
+ Multivariate_ EOF analysis
+ Significance analysis via bootstrapping
+ Maximum Covariance Analysis

.. _T-mode: https://xeofs.readthedocs.io/en/latest/auto_examples/1eof/plot_eof-tmode.html#sphx-glr-auto-examples-1eof-plot-eof-tmode-py
.. _S-mode: https://xeofs.readthedocs.io/en/latest/auto_examples/1eof/plot_eof-smode.html#sphx-glr-auto-examples-1eof-plot-eof-smode-py
.. _Weighted: https://xeofs.readthedocs.io/en/latest/auto_examples/1eof/plot_weighted_eof.html#sphx-glr-auto-examples-1eof-plot-weighted-eof-py
.. _Rotated: https://xeofs.readthedocs.io/en/latest/auto_examples/1eof/plot_rotated_eof.html#sphx-glr-auto-examples-1eof-plot-rotated-eof-py
.. _Multivariate: https://xeofs.readthedocs.io/en/latest/auto_examples/1eof/plot_multivariate-eof-analysis.html#sphx-glr-auto-examples-1eof-plot-multivariate-eof-analysis-py
.. _Documentation: https://xeofs.readthedocs.io/en/latest/
.. _examples: https://xeofs.readthedocs.io/en/latest/auto_examples/index.html



Credits
----------------------

- to Andrew Dawson_ for the first and fundamental Python package for EOF analysis
- to Yefee_ from which I took some inspiration to implement MCA
- to James Chapman_ who created a great Python package for Canonical Correlation Analysis
- to Diego Bueso_ for his open-source ROCK-PCA implementation in Matlab
- to yngvem_ for how to organize the project folder structure
- to all the developers of NumPy_, pandas_ \& xarray_ for their invaluable contributions to science


.. _NumPy: https://www.numpy.org
.. _pandas: https://pandas.pydata.org
.. _xarray: https://xarray.pydata.org
.. _yngvem: https://github.com/yngvem/python-project-structure
.. _pooch: https://github.com/fatiando/pooch
.. _Chapman: https://github.com/jameschapman19/cca_zoo
.. _Bueso: https://github.com/DiegoBueso/ROCK-PCA
.. _Dawson: https://github.com/ajdawson/eofs
.. _Yefee: https://github.com/Yefee/xMCA


How to cite?
----------------------
Please make sure that when using ``xeofs`` you always cite the **original source** of the method used. Additionally, if you find ``xeofs`` useful for your research, you may cite it as follows::

   @software{rieger_xeofs_2022,
     title = {xeofs: Multi-dimensional {EOF} analysis and variants in xarray},
     url = {https://github.com/nicrie/xeofs}
     version = {0.6.0},
     author = {Rieger, Niclas},
     date = {2022},
     doi = {10.5281/zenodo.6323011}
   }
