.. _api:

======================
API
======================

.. warning:: 

    The package is under development, and its API may change.

Single-Set Analysis
====================

Methods that examine relationships among variables within a single dataset, or when multiple datasets are combined and analyzed as one.

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   xeofs.models.EOF
   xeofs.models.EOFRotator
   xeofs.models.ComplexEOF
   xeofs.models.ComplexEOFRotator
   xeofs.models.ExtendedEOF
   xeofs.models.OPA
   xeofs.models.GWPCA


Multi-Set Analysis
==================
Methods that investigate relationships or patterns between variables across two or more distinct datasets.

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   xeofs.models.MCA
   xeofs.models.ComplexMCA
   xeofs.models.MCARotator
   xeofs.models.ComplexMCARotator
   xeofs.models.CCA


Model Evaluation
================
Tools to assess the quality of your model.

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   xeofs.validation.EOFBootstrapper


Utilities
=========
Support functions.

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   xeofs.models.RotatorFactory









