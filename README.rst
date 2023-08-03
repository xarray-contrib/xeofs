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

.. |build| image:: https://img.shields.io/github/actions/workflow/status/nicrie/xeofs/CI.yml?branch=main
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

``xeofs`` is a Python package designed for Empirical Orthogonal Function (EOF) analysis, also known as Principal Component Analysis (PCA), 
and related variants. The package stands out due to its capacity 
to handle multi-dimensional Earth observation data, thereby optimizing the speed and efficiency of EOF analysis. 
Here are the key strengths of ``xeofs``:

The benefits of using ``xeofs`` include:

- **Multi-dimensional Analysis**: Execute labeled EOF analysis with the extensive features of ``xarray``.
- **Scalability**: Handle large datasets effectively with ``dask``.
- **Speed**: Enjoy quick EOF analysis using ``scipy``'s randomized SVD.
- **Variety of Methods**: Perform diverse variants of EOF analysis, including complex and rotated version, along with related techniques such as Maximum Covariance Analysis (MCA).
- **Model Validation**: Validate models through bootstrapping.
- **Modular Code Structure**: Incorporate new EOF variants with ease due to the package's modular structure.
- **Flexible Data Formats**: Accepts a variety of ``xarray`` input types (``DataArray``, ``Dataset``, list of ``DataArray``).

Compared to similar packages like eofs_, pyEOF_, and xMCA_, ``xeofs`` is more comprehensive and flexible, providing unique capabilities like handling fully multidimensional dimensions 
(both samples and features) and a simple interface for bootstrapping. This makes ``xeofs`` a powerful one-stop-shop for most of your EOF analysis needs.

.. _pyEOF: https://github.com/zhonghua-zheng/pyEOF
.. _xMCA: https://github.com/Yefee/xMCA
.. _eofs: https://github.com/ajdawson/eofs

Installation
------------

To install the package, use either of the following commands:

.. code-block:: bash

   pip install xeofs

or 

.. code-block:: bash

   conda install -c conda-forge xeofs


Quickstart
----------

In order to get started with ``xeofs``, follow these simple steps:

1. **Import the package**

   .. code-block:: python

      import xeofs as xe

2. **Instantiate the model**

   Select the type of analysis you want to perform (in this case, EOF analysis) and set the parameters. For example, if you want to analyze the first 10 modes, you would use the following code:

   .. code-block:: python

      model = xe.models.EOF(n_modes=10)

3. **Fit the model to your data**

   Fit the model to your data by specifying the dimensions along which the analysis should be performed. Replace 'your_data' and 'your_dimension' with your specific data and dimension:

   .. code-block:: python

      model.fit(your_data, dim=your_dimension)

Congratulations! You have performed your first analysis with ``xeofs``. To further explore the capabilities of ``xeofs``, check the documentation_ and examples_.



Documentation
-------------

For a more comprehensive overview and usage examples, visit the documentation_.

Contributing
------------

Contributions are highly welcomed and appreciated. If you're interested in improving ``xeofs`` or fixing issues, please open a Github issue_.

License
-------

This project is licensed under the terms of the MIT license.

Contact
-------

For questions or support, please open a Github issue_.



.. _issue: https://github.com/nicrie/xeofs/issues
.. _documentation: https://xeofs.readthedocs.io/en/latest/
.. _examples: https://xeofs.readthedocs.io/en/latest/auto_examples/index.html



Credits
----------------------

I want to acknowledge

- Andrew Dawson_, for his foundational Python package for EOF analysis.
- Yefee_, whose work provided useful references for implementing MCA in ``xeofs``.
- James Chapman_, creator of a Python package for Canonical Correlation Analysis.
- Diego Bueso_, for his open-source ROCK-PCA implementation in Matlab.
- The developers of NumPy_, pandas_, and xarray_ for their indispensable tools for scientific computations in Python.



.. _NumPy: https://www.numpy.org
.. _pandas: https://pandas.pydata.org
.. _xarray: https://xarray.pydata.org
.. _Chapman: https://github.com/jameschapman19/cca_zoo
.. _Bueso: https://github.com/DiegoBueso/ROCK-PCA
.. _Dawson: https://github.com/ajdawson/eofs
.. _Yefee: https://github.com/Yefee/xMCA


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
