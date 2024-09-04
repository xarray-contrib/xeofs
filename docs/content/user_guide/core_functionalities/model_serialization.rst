=============================================
Model Serialization
=============================================

xeofs models offer convenient ``save()`` and ``load()`` methods for serializing
fitted models to a portable format. 

.. code-block:: python

  from xeofs.single import EOF

  model = EOF()
  model.fit(data, dim="time")
  model.save("my_model.zarr")

  # Later, you can load the model
  loaded_model = EOF.load("my_model.zarr")


