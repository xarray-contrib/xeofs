======================
API Reference
======================

.. warning:: 

   The package is under development, and its API may change.

The xeofs package focuses on eigenmethods for dimensionality reduction in climate science. These methods are categorized into two groups:

1. :doc:`Single-Set Analysis <single_set_analysis>`: Methods that examine relationships or patterns within a single dataset.
2. :doc:`Multi-Set Analysis <multi_set_analysis>`: Methods that investigate relationships or patterns between variables across two or more distinct datasets.

--------------------
Single-Set Analysis
--------------------

A classic example of single-set analysis is Principal Component Analysis (PCA/EOF analysis), used to extract the dominant patterns of variability within a single dataset. While PCA can be applied to multiple (standardized) datasets simultaneously, it treats all datasets as one large dataset, maximizing overall variability without considering inter-dataset relationships. Consequently, the most important variables may come from only one dataset, ignoring others.

--------------------
Multi-Set Analysis
--------------------

Examples of multi-set analysis methods include Canonical Correlation Analysis (CCA) and Maximum Covariance Analysis (MCA). These techniques identify shared patterns of variability between two or more datasets, focusing on common patterns rather than those unique to each dataset. For instance, if you have two datasets (e.g., monthly temperatures from tropical and polar regions over 70 years), CCA or MCA would likely highlight the global warming signal as the dominant pattern common to both datasets, while the seasonal cycle would not be dominant as it is only prominent in the polar region.

Additionally, xeofs offers tools for :doc:`model evaluation <model_evaluation>`, though these are still in early development stages.



.. toctree::
   :maxdepth: 3
   :hidden:

   single_set_analysis
   multi_set_analysis
   model_evaluation
   utilities

