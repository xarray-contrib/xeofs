===================
Migrating to ``v3``
===================

This short guide outlines the key changes in version 3 of ``xeofs`` and provides instructions for updating your code from version 2.

-----------------------------
Updated Namespaces for Models
-----------------------------

To improve organization and future scalability, :doc:`new namespaces <../api_reference/index>` have been introduced for the models. This is a breaking change, but updating your code is straightforward. You'll need to adjust the import statements and method access in your v2 code. The new namespaces are as follows:

- ``xeofs.single``: Contains all single-field models (e.g. EOF analysis)
- ``xeofs.cross``: Contains all cross-field models (e.g. MCA)
- ``xeofs.multi``: Contains all multi-field models 

Previously, methods were accessed like this:

.. code-block:: python

    eof = xe.models.EOF()

In ``v3``, they are now accessed as follows:

.. code-block:: python

    eof = xe.single.EOF()
    mca = xe.cross.MCA()

--------------------------------
Complex/Hilbert Classes Reanming
--------------------------------
In ``v2``, Hilbert methods were mistakenly named as Complex methods. This has been corrected in ``v3``. If you were using Complex methods in ``v2``, you should now use Hilbert methods in ``v3``.


.. code-block:: python

    # v2
    ceof = xe.models.ComplexEOF()

    # v3
    ceof = xe.single.HilbertEOF()


Complex methods still exist but now follow the standard convention of analyzing pre-existing complex fields (e.g., wind field components ``u`` and ``v``). Hilbert methods, on the other hand, are used for analyzing real physical fields, where the Hilbert transform is applied to convert the real field into a complex field.

------------------------------------
Normalization of ``score()`` Methods
------------------------------------
To align with the scikit-learn API, the default value for the ``normalize`` parameter in the ``score()`` methods has been changed from ``True`` to ``False``. In EOF analysis, this means that the raw principal components (PCs) will now be returned, with each PC weighted by its respective mode's importance (singular value). If you prefer the previous behavior, you can explicitly set ``normalize=True``.

The ``components()`` method remains unchanged, with normalization still set to ``True`` by default.

------------------
Dropped Parameters
------------------
Several parameters have been removed in ``v3``:

- ``verbose``: This parameter has been removed from all methods, as it was deemed unnecessary
- ``squared_loadings`` (specific to ``MCARotator`` classes): This parameter has been removed because, although it preserved the squared covariance fraction after rotation, it invalidated the ``inverse_transform`` method for reconstructing the original fields. Its removal helps avoid confusion and misuse.

--------------------------
Standardize Variable Names
--------------------------
In ``v3``, variable names have been standardized to ``X`` and ``Y`` for input data. In ``v2``, input data was often referred to as ``data``, ``data1``, or ``data2``. This change improves code readability and aligns with the scikit-learn API for consistency.

