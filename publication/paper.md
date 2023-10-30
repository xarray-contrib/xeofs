# Meta data file

generate a [meta data file](https://gist.github.com/arfon/478b2ed49e11f984d6fb) and add it to the repo

---
title: 'xeofs: Dimensionality reduction in xarray'
tags:
  - Python
  - climate science
  - xarray
  - dask
  - dimensionality reduction
  - EOF analysis
  - PCA
authors:
  - name: Niclas Rieger
    orcid: 0000-0003-3357-1742
    affiliation: "1, 2, 3"
  - name: Samuel J. Levang
    affiliation: 3
  - name: Aaron Spring
    affiliation: 4
affiliations:
  - name: Centre de Recerca Matemàtica (CRM), Bellaterra, Spain
    index: 1
  - name: Departament de Física, Universitat Autònoma de Barcelona, Bellaterra, Spain
    index: 2
  - name: Instituto de Ciencias del Mar (ICM) - CSIC, Barcelona, Spain
    index: 3
  - name: TBF
    index: 4
  - name: TBF
    index: 5
date: 30 October 2023
bibliography: paper.bib

---

# Summary

`xeofs` is a Python package tailored for the climate science 
community, designed to streamline advanced data analysis using dimensionality 
reduction techniques like Empirical Orthogonal Functions (EOF) analysis-often
called Principal Component Analysis (PCA) in other domains. 
Integrating seamlessly with `xarray` objects [@hoyer2017xarray], `xeofs` 
makes it easier to analyze large, labeled, multi-dimensional datasets.
By harnessing `Dask`'s capabilities [@dask_2016], it scales computations efficiently
across multiple cores or clusters, apt for extensive climate data applications.


# Statement of Need

Climate science routinely deals with analyzing large, multi-dimensional datasets that 
are dense with complex physical process information. The extraction of meaningful 
insights from such vast datasets is challenging and often requires the application
of dimensionality reduction techniques like EOF analysis (PCA outside climate science). 
Packages such as `scikit-learn` [@pedregosa_scikit-learn_2011]
offer a range of reduction techniques, yet they often fall short of meeting the 
specific needs of climate scientists who work with variants of PCA [@@hannachi_patterns_2021] 
including ROCK-PCA [@bueso_nonlinear_2020] and spectral, rotated PCA [@guilloteau_rotated_2020].

Climate datasets are inherently multi-dimensional, usually involving time, longitude
and latitude, and often include missing values representing geographical features like 
oceans or land. These characteristics require meticulous data transformations and tracking 
of missing values and dimension coordinates, which can be cumbersome and prone to error, 
increasing the workload, especially for smaller-scale projects. Furthermore, the size 
of climate datasets often necessitates out-of-memory processing.

While `xMCA` [@xmca_yefee] and `eofs` [@dawson2016eofs] have addressed some of these 
issues by offering analysis tools compatible with `xarray` and `Dask`, `xeofs` 
expands on these by including a broader range of techniques such as 
rotated [@kaiser_varimax_1958], complex/Hilbert [@rasmusson_biennial_1981], and 
extended [@weare_examples_1982] PCA/EOF analysis. `xeofs` operates natively 
with `xarray` objects, preserving data labels and structure, and handles 
datasets with missing values adeptly. It also integrates seamlessly with `Dask`
and shows improved performance in particular for larger datasets 
\autoref{fig:computation_times} due to its usage of randomized 
Singular Value Decomposition (SVD) [@halko_randomized_2011].

![Comparison of computation times of PCA for varying number of features between `xeofs` and `eofs`.\label{fig:computation_times}](../docs/img/timings_light.png){ width=100% }


# Implementation
Methods in `xeofs` are implemented in a way that they loosly follow `scikit-learn`
conventions, providing a user-friendly interface with a each method being a class
with a `fit` method. In addition, when appropriate a `transform` and `inverse_transform` 
method are provided. Furthermore, `xeofs` allows the user to easily add their own
dimensionality reduction techniques by providing a low-level entry point to the
internal processing pipeline.
Finally, `xeofs` provides a bootstrapping module for model evaluation which currently
supports a straightforward way to bootstrap a PCA model.

# Available Methods

At the time of publication, `xeofs` provides the following methods:

| Method                        | Alternative name                                                | Reference                                                                             |
| :---                          | :---                                                            |     :---:                                                                             |
| PCA                           | EOF analysis                                                    |                                                                                       |
| Rotated PCA                   | -                                                               | [@kaiser_varimax_1958; @hendrickson_promax_1964]                                      |
| Complex PCA                   | Hilbert EOF (HEOF) analysis                                     | [@rasmusson_biennial_1981; @barnett_interaction_1983; @horel_complex_1984]            |
| Complex Rotated PCA           | -                                                               | [@horel_complex_1984]                                                                 |
| Extended PCA                  | EEOF analysis / Multichannel Singular Spectrum Analysis (M-SSA) | [@weare_examples_1982; broomhead_extracting_1986]                                     |
| Optimal Persistence Analysis  | OPA                                                             | [@delsole_optimally_2001; @delsole_low-frequency_2006]                                |
| Geographically-Weighted PCA   | GWPCA                                                           | [@harris_geographically_2011]                                                         |
| Maximum Covariance Analysis   | MCA, SVD analysis                                               | [@bretherton_intercomparison_1992]                                                    |
| Rotated MCA                   | -                                                               | [@cheng_orthogonal_1995]                                                              |
| Complex MCA                   | Hilbert MCA/Analytical SVD                                      | [@elipot_observed_2017]                                                               |
| Complex Rotated MCA           | -                                                               | [@rieger_lagged_2021]                                                                 |
| Canonical Correlation Analysis| CCA                                                             | [@hotelling_relations_1936; @vinod_canonical_1976; @bretherton_intercomparison_1992]  |

However, we note that we plan to implement more methods in the future, including ROCK-PCA and spectral, rotated PCA.



# Acknowledgements
We extend our gratitude to those who have improved the software by reporting issues and providing feedback.

This work contributes to the Climate Advanced Forecasting of sub-seasonal Extremes (CAFE) project and was conducted as part of the Physics doctoral program at the Autonomous University of Barcelona. NR acknowledges the European Union’s Horizon 2020 research and innovation program for funding under the Marie Skłodowska-Curie grant agreement No 813844.


# References