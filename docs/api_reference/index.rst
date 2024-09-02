======================
API Reference
======================

.. warning:: 

   The package is under development, and its API may change.

The xeofs package focuses on eigenmethods for dimensionality reduction in climate science. It is organized into methods that examine relationships between variables 

1. within a **single dataset** (``xeofs.single``),
2. across **two datasets**  (``xeofs.cross``) and
3. across **more than two datasets** (``xeofs.multi``).

--------------------
Single-Set Analysis
--------------------

A classic example of :doc:`single-set analysis <single_set_analysis>` is Principal Component Analysis (PCA/EOF analysis), used to extract the dominant patterns of variability within a single dataset. While PCA can be applied to multiple (standardized) datasets simultaneously, it treats all datasets as one large dataset, maximizing overall variability without considering inter-dataset relationships. Consequently, the most important variables may come from only one dataset, ignoring others.

----------------------------
Cross and Multi-Set Analysis
----------------------------

Classic examples of :doc:`cross <cross_set_analysis>` or :doc:`multi<multi_set_analysis>`-set analysis methods include Canonical Correlation Analysis (CCA), Maximum Covariance Analysis (MCA) and Redundancy Analyis (RDA). These techniques identify shared patterns of variability between two distinct datasets, focusing on common patterns rather than those unique to each dataset. 


Additionally, xeofs offers tools for :doc:`model evaluation <model_evaluation>`, though these are still in early development stages.



.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Methods

   single_set_analysis
   cross_set_analysis
   multi_set_analysis
   utilities

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Significance Testing

   model_evaluation

