.. image:: docs/logos/xeofs_logo.png
  :align: center
  :width: 800
  :alt: xeofs logo


+----------------------------+-----------------------------------------------------+
| Versions                   | |pypi| |conda|                                      |
+----------------------------+-----------------------------------------------------+
| Build & Testing            | |build| |coverage|                                  |
+----------------------------+-----------------------------------------------------+
| Code Quality               | |black|                                             |
+----------------------------+-----------------------------------------------------+
| Documentation              | |docs|                                              |
+----------------------------+-----------------------------------------------------+
| Citation & Licensing       | |zenodo| |license|                                  |
+----------------------------+-----------------------------------------------------+
| User Engagement            | |downloads|                                         |
+----------------------------+-----------------------------------------------------+

.. |pypi| image:: https://img.shields.io/pypi/v/xeofs
   :target: https://pypi.org/project/xeofs/
   :alt: Python Package Index (PyPI)

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/xeofs
   :target: https://anaconda.org/conda-forge/xeofs
   :alt: Conda-forge Version

.. |build| image:: https://img.shields.io/github/actions/workflow/status/nicrie/xeofs/ci.yml?branch=main
   :target: https://github.com/nicrie/xeofs/actions
   :alt: Build Status

.. |docs| image:: https://readthedocs.org/projects/xeofs/badge/?version=latest
   :target: https://xeofs.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black Code Style

.. |coverage| image:: https://codecov.io/gh/nicrie/xeofs/branch/main/graph/badge.svg?token=8040ZDH6U7
    :target: https://codecov.io/gh/nicrie/xeofs
    :alt: Test Coverage

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.6323012.svg
   :target: https://doi.org/10.5281/zenodo.6323012
   :alt: DOI - Zenodo

.. |license| image:: https://img.shields.io/pypi/l/xeofs
   :target: https://github.com/nicrie/xeofs/blob/main/LICENSE
   :alt: License

.. |downloads| image:: https://img.shields.io/pypi/dw/xeofs
   :alt: PyPI - Downloads



Overview
---------------------

``xeofs`` is a dedicated Python package for dimensionality reduction in the realm of climate science, 
offering methods like PCA, known as EOF analysis within the field, and related variants. 
Seamlessly integrated with ``xarray`` and ``Dask``, it's tailored for easy handling and scalable 
computation on large, multi-dimensional datasets, making advanced climate data analysis both accessible and efficient.

- **Multi-Dimensional**: Designed for ``xarray`` objects, it applies dimensionality reduction to multi-dimensional data while maintaining data labels.
- **Dask-Integrated**: Supports large datasets via ``Dask`` xarray objects
- **Extensive Methods**: Offers various dimensionality reduction techniques
- **Adaptable Output**: Provides output corresponding to the type of input, whether single or list of ``xr.DataArray`` or ``xr.Dataset``
- **Missing Values**: Handles ``NaN`` values within the data
- **Bootstrapping**: Comes with a user-friendly interface for model evaluation using bootstrapping
- **Efficient**: Ensures computational efficiency, particularly with large datasets through randomized SVD
- **Modular**: Allows users to implement and incorporate new dimensionality reduction methods



.. _pyEOF: https://github.com/zhonghua-zheng/pyEOF
.. _xMCA: https://github.com/Yefee/xMCA
.. _eofs: https://github.com/ajdawson/eofs

Installation
------------

To install the package, use either of the following commands:

.. code-block:: bash

   conda install -c conda-forge xeofs

or 

.. code-block:: bash

   pip install xeofs

Quickstart
----------

In order to get started with ``xeofs``, follow these simple steps:

**Import the package**

.. code-block:: python

   import xeofs as xe

**EOF analysis**

.. code-block:: python

   model = xe.models.EOF(n_modes=10)
   model.fit(data, dim="time")
   comps = model.components()  # EOFs (spatial patterns)
   scores = model.scores()  # PCs (temporal patterns)

**Varimax-rotated EOF analysis**

.. code-block:: python

   rotator = xe.models.EOFRotator(n_modes=10)
   rotator.fit(model)
   rot_comps = rotator.components()  # Rotated EOFs (spatial patterns)
   rot_scores = rotator.scores()  # Rotated PCs (temporal patterns)

**MCA**

.. code-block:: python

   model = xe.models.MCA(n_modes=10)
   model.fit(data1, data2, dim="time")
   comps1, comps2 = model.components()  # Singular vectors (spatial patterns)
   scores1, scores2 = model.scores()  # Expansion coefficients (temporal patterns)

**Varimax-rotated MCA**

.. code-block:: python

   rotator = xe.models.MCARotator(n_modes=10)
   rotator.fit(model)
   rot_comps = rotator.components()  # Rotated singular vectors (spatial patterns)
   rot_scores = rotator.scores()  # Rotated expansion coefficients (temporal patterns)


To further explore the capabilities of ``xeofs``, check out the available documentation_ and examples_.
For a full list of currently available methods, see the `Reference API`_.

.. _`Reference API`: https://xeofs.readthedocs.io/en/latest/api.html



Documentation
-------------

For a more comprehensive overview and usage examples, visit the documentation_.

Contributing
------------

Contributions are highly welcomed and appreciated. If you're interested in improving ``xeofs`` or fixing issues, please read our 
`Contributing Guide`_.

License
-------

This project is licensed under the terms of the MIT_ license.

Contact
-------

For questions or support, please open a Github issue_.



.. _issue: https://github.com/nicrie/xeofs/issues
.. _`Contributing Guide`: https://xeofs.readthedocs.io/en/latest/overview_3_contributing.html
.. _documentation: https://xeofs.readthedocs.io/en/latest/
.. _examples: https://xeofs.readthedocs.io/en/latest/auto_examples/index.html
.. _MIT: https://github.com/nicrie/xeofs/blob/main/LICENSE


Credits
----------------------

- Randomized PCA: scikit-learn_
- EOF analysis: Python package eofs_ by Andrew Dawson
- MCA: Python package xMCA_ by Yefee
- CCA: Python package CCA-Zoo_ by James Chapman
- ROCK-PCA: Matlab implementation by Diego Bueso_



.. _NumPy: https://www.numpy.org
.. _pandas: https://pandas.pydata.org
.. _xarray: https://xarray.pydata.org
.. _scikit-learn: https://scikit-learn.org/stable/
.. _CCA-Zoo: https://github.com/jameschapman19/cca_zoo
.. _Bueso: https://github.com/DiegoBueso/ROCK-PCA
.. _eofs: https://github.com/ajdawson/eofs
.. _xMCA: https://github.com/Yefee/xMCA


How to cite?
----------------------
When utilizing ``xeofs``, kindly remember to cite the original creators of the methods employed in your work. Additionally, if ``xeofs`` is proving useful in your research, I'd appreciate if you could acknowledge its use with the following citation:

.. code-block:: bibtex

   @software{rieger_xeofs_2023,
     title = {xeofs: Multi-dimensional {EOF} analysis and variants in xarray},
     url = {https://github.com/nicrie/xeofs}
     version = {1.x.y},
     author = {Rieger, Niclas},
     date = {2023},
     doi = {10.5281/zenodo.6323011}
   }
