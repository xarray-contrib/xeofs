:sd_hide_title:

.. image:: logos/xeofs_logo.png
  :align: center
  :width: 800
  :alt: logo of xeofs

#############################################################
Advanced EOF Analysis in Python
#############################################################
``xeofs`` is a toolbox designed to handle multi-dimensional EOF analysis and related methods
for Earth system sciences, leveraging the power of packages like ``xarray`` for 
labeled, multi-dimensional analysis and 
``dask`` for scalability. It supports various forms of 
EOF analysis including standard, rotated, and multivariate analysis. 
The overall goal of ``xeofs`` is to unify and extend existing EOF implementations, 
thereby enhancing their application in the broader scientific community.


.. grid:: 2

    .. grid-item-card::  Overview
      :link: overview
      :link-type: doc


      Get an idea of what xeofs is all about.

    .. grid-item-card::  Getting started
      :link: getting_started
      :link-type: doc

      Learn how to use xeofs.


.. grid:: 2

    .. grid-item-card::  Examples
      :link: auto_examples/index
      :link-type: doc

      Follow along with some examples. 

    .. grid-item-card::  API
      :link: api
      :link-type: doc

      Explore the API.


.. note:: Work in progress.



.. toctree::
   :maxdepth: 3
   :hidden:

   overview
   installation
   getting_started
   auto_examples/index
   api
