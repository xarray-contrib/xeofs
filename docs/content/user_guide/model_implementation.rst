=============================================
Implement Your Own Model
=============================================

The xeofs package has been designed with modularity in mind, allowing you to seamlessly incorporate new methods. 
For instance, if you'd like to introduce a new dimensionality reduction technique named ``MyModel``, 
you can achieve this by inheriting of either the ``BaseModelSingleSet`` or ``BaseModelCrossSet`` class and 
implementing its ``_fit_algorithm()`` method.

Here's a detailed walkthrough on how to incorporate a new model:

--------------------------------------------
1. Inherit the BaseModel
--------------------------------------------
    
Your new model should inherit from the `BaseModel` class. This abstract base class enables 
the transformation of any ND xarray object into a 2D ``xarray.DataArray`` with dimensions 
(sample, feature) and back. Additionally, it grants access to handy parameters like 
``n_modes``, ``standardize``, and ``use_coslat``.

.. code-block:: python

  from xeofs.single.base_model_single_set import BaseModelSingleSet
  from xeofs.linalg.decomposer import Decomposer

  class MyModel(BaseModelSingleSet):
      def __init__(self, **kwargs):
          super().__init__(**kwargs)

--------------------------------------------
2. Define the Fit Algorithm
--------------------------------------------
    
Your chosen method's entire operation should be encapsulated within the 
``_fit_algorithm()``. This function is triggered by ``fit()`` and handles the model fitting. 
By this stage, xeofs has already processed essential preprocessing steps, ranging from 
centering and weighting to stacking and handling ``NaN`` values.

Here's a basic PCA example to illustrate the process:

.. code-block:: python

  def _fit_algorithm(self, data):
      # NOTE: The `data` here is a 2D xarray.DataArray with dimensions (sample, feature).

      # We'll illustrate with a simplified PCA.
      # The goal is to perform an SVD on the `data` matrix.
      decomposer = Decomposer(n_modes=self.n_modes)
      decomposer.fit(data)

      # Extract the necessary components from the decomposer.
      scores = decomposer.U_
      components = decomposer.V_
      singular_values = decomposer.s_

      # Store the data for later access using the internal DataContainer class.
      self.data.add(name="my_singular_values", data=singular_values)
      self.data.add(name="my_components", data=components)
      self.data.add(name="my_scores", data=scores)

      # (Optional) Attach model parameters as attributes to your data.
      self.data.set_attrs(self.attrs)

--------------------------------------------
3. Access the Results
--------------------------------------------
    
After fitting your model, results can be retrieved by creating a method for each data 
piece. The internal ``Preprocessor`` class can assist with this task, 
ensuring that the retrieved data retains the correct format.

Depending on their dimensions, data types are categorized into four groups:

1. (sample, feature, ...)
2. (sample, ...)
3. (feature, ...)
4. (...)

The `Preprocessor` class offers methods corresponding to the first three data groups:

- ``inverse_transform_data`` for (sample, feature, ...)
- ``inverse_transform_scores`` for (sample, ...)
- ``inverse_transform_components`` for (feature, ...)

For group (4), data can be accessed directly since there's no need for back transformation.

.. code-block:: python

    def my_singular_values(self):
        return self.data.get("my_singular_values")

    def my_components(self):
        return self.preprocessor.inverse_transform_components(
            self.data.get("my_components")
        )

    def my_scores(self):
        return self.preprocessor.inverse_transform_scores(self.data.get("my_scores"))


----------------------------------------------------------------------------------------
4. Optional: Implement Transform and Inverse Transform Methods
----------------------------------------------------------------------------------------

While it's required to implement the ``transform`` and ``inverse_transform`` methods for a complete model, 
we'll merely indicate their absence for this example.

.. code-block:: python

  def _transform_algorithm(self, data):
      raise NotImplementedError("This model does not support transform.")

  def _inverse_transform_algorithm(self, scores):
      raise NotImplementedError("This model does not support inverse transform.")

--------------------------------------------
5. Execute the Model
--------------------------------------------

With all parts in place, you can now initialize and use the new model:

.. code-block:: python

    model = MyModel(n_modes=3)
    model.fit(t2m, dim="time")
    model.my_components()

