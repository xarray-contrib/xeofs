
.. image:: logos/xeofs_logo_dark.png
  :class: only-dark
  :align: center
  :width: 1000
  :alt: logo of xeofs

|
.. image:: logos/xeofs_logo_light.png
  :class: only-light
  :align: center
  :width: 1000
  :alt: logo of xeofs

|

.. rst-class:: center

=====================================================
Extracting Patterns from Climate Data
=====================================================

.. toctree::
  :maxdepth: 3
  :hidden:

  user_guide/index
  installation
  api_reference/index
  whats_new/CHANGELOG
  contributing_guide



xeofs is a specialized Python package designed for dimensionality reduction in climate science, aimed at extracting meaningful patterns from large datasets. It provides eigenmethods such as Principal Component Analysis (EOF analysis) and several related variants. Seamlessly integrated with xarray_ and Dask_, xeofs enables efficient handling and scalable computation of large, multi-dimensional datasets. This integration makes advanced climate data analysis both accessible and efficient.

.. grid:: 2

    .. grid-item-card:: 
      :octicon:`repo;10em`
      :link: user_guide/index
      :link-type: doc
      :text-align: center

      **User Guide**

      Learn more about the package and its features.

    .. grid-item-card::
      :octicon:`gear;10em`
      :link: installation
      :link-type: doc
      :text-align: center

      **Installation**

      Get started with xeofs in a few simple steps.


.. grid:: 2

    .. grid-item-card::
      :octicon:`search;10em`
      :link: api_reference/index
      :link-type: doc
      :text-align: center

      **API Reference**

      Explore the available functions and classes.

    .. grid-item-card::
      :octicon:`people;10em`
      :link: contributing_guide
      :link-type: doc
      :text-align: center

      **Contributing Guide**

      Join the community and contribute to the project.


.. note:: 

    xeofs is currently in the early stages of development. 
    We welcome contributions and feedback from the community.

.. _xarray: https://docs.xarray.dev/en/stable/index.html
.. _Dask: https://dask.org/


