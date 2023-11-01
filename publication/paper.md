---
title: 'xeofs: Comprehensive EOF analysis in Python with xarray'
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
    affiliation: 4
affiliations:
  - name: Centre de Recerca Matemàtica (CRM), Bellaterra, Spain
    index: 1
  - name: Departament de Física, Universitat Autònoma de Barcelona, Bellaterra, Spain
    index: 2
  - name: Instituto de Ciencias del Mar (ICM) - CSIC, Barcelona, Spain
    index: 3
  - name: Salient Predictions, Cambridge, MA, USA
    index: 4
date: 1 November 2023
bibliography: paper.bib

---

# Summary

`xeofs` is a Python package tailored for the climate science 
community, designed to streamline advanced data analysis using dimensionality 
reduction techniques like Empirical Orthogonal Functions (EOF) analysis-often
called Principal Component Analysis (PCA) in other domains. 
Integrating seamlessly with `xarray` objects [@hoyer2017xarray], `xeofs` 
makes it easier to analyze large, labeled, multi-dimensional datasets.
By harnessing `Dask`'s capabilities [@dask2016], it scales computations efficiently
across multiple cores or clusters, apt for extensive climate data applications.


# Statement of Need
Climate science routinely deals with analyzing large, multi-dimensional datasets,
whose complexity mirrors the intricate dynamics of the climate system itself. The extraction of meaningful 
insights from such vast datasets is challenging and often requires the application
of dimensionality reduction techniques like EOF analysis (PCA outside climate science). 
Packages such as `scikit-learn` [@pedregosa_scikit-learn_2011]
offer a range of reduction techniques, yet they often fall short of meeting the 
specific needs of climate scientists who work with variants of PCA [@hannachi_patterns_2021] 
including ROCK-PCA [@bueso_nonlinear_2020] and spectral, rotated PCA [@guilloteau_rotated_2020].

Climate datasets are inherently multi-dimensional, usually involving time, longitude
and latitude, and often include missing values representing geographical features like 
oceans or land. These characteristics require meticulous data transformations and tracking 
of missing values and dimension coordinates, which can be cumbersome and prone to error, 
increasing the workload, especially for smaller-scale projects. Furthermore, the size 
of climate datasets often necessitates out-of-memory processing.

While `xMCA` [@xmca_yefee] and `eofs` [@dawson_eofs_2016] have addressed some of these 
issues by offering analysis tools compatible with `xarray` and `Dask`, `xeofs` 
expands on these by including a broader range of techniques such as 
rotated [@kaiser_varimax_1958], complex/Hilbert [@rasmusson_biennial_1981], and 
extended [@weare_examples_1982] PCA/EOF analysis. `xeofs` operates natively 
with `xarray` objects, preserving data labels and structure, and handles 
datasets with missing values adeptly. It also integrates seamlessly with `Dask`
and shows improved performance in particular for larger datasets 
(\autoref{fig:computation_times}) due to its usage of randomized 
Singular Value Decomposition (SVD) [@halko_finding_2011].

![(A) Evaluation of xeofs computation times for processing 3D data sets of varying sizes. (B) Performance comparison between `xeofs` and `eofs` across different data set dimensions. Dashed black line indicates the contour of datasets approximately 3 MiB in size. Tests conducted [^1] on an Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz, 12 threads (6 cores), with 16GB DDR4 RAM at 2667 MT/s. \label{fig:computation_times}](../docs/perf/timings_light.png){ width=100% }

[^1]: The script used to generate these results is available at https://github.com/nicrie/xeofs/blob/main/docs/perf/ .

# Implementation
`xeofs` adopts the familiar `scikit-learn` style, delivering an intuitive interface 
where each method is a class with `fit`, and when applicable, `transform` 
and `inverse_transform` methods. It also offers flexibility by allowing users to 
introduce custom dimensionality reduction methods via a streamlined entry point 
to its internal pipeline. Additionally, the package includes a bootstrapping 
module for straightforward PCA model evaluation.

# Available Methods

At the time of publication, `xeofs` provides the following methods:

| Method                        | Alternative name                                                | Reference                                                                             |
| :---                          | :---                                                            |     :---:                                                                             |
| PCA                           | EOF analysis                                                    |                                                                                       |
| Rotated PCA                   | -                                                               | [@kaiser_varimax_1958; @hendrickson_promax_1964]                                      |
| Complex PCA                   | Hilbert EOF (HEOF) analysis                                     | [@rasmusson_biennial_1981; @barnett_interaction_1983; @horel_complex_1984]            |
| Complex Rotated PCA           | -                                                               | [@horel_complex_1984]                                                                 |
| Extended PCA                  | EEOF analysis / Multichannel Singular Spectrum Analysis (M-SSA) | [@weare_examples_1982; @broomhead_extracting_1986]                                     |
| Optimal Persistence Analysis  | OPA                                                             | [@delsole_optimally_2001; @delsole_low-frequency_2006]                                |
| Geographically-Weighted PCA   | GWPCA                                                           | [@harris_geographically_2011]                                                         |
| Maximum Covariance Analysis   | MCA, SVD analysis                                               | [@bretherton_intercomparison_1992]                                                    |
| Rotated MCA                   | -                                                               | [@cheng_orthogonal_1995]                                                              |
| Complex MCA                   | Hilbert MCA/Analytical SVD                                      | [@elipot_observed_2017]                                                               |
| Complex Rotated MCA           | -                                                               | [@rieger_lagged_2021]                                                                 |
| Canonical Correlation Analysis| CCA                                                             | [@hotelling_relations_1936; @vinod_canonical_1976; @bretherton_intercomparison_1992]  |

Additionally, we are actively developing further enhancements to `xeofs`, with plans to incorporate advanced methods 
such as ROCK-PCA [@bueso_nonlinear_2020] and spectral, rotated PCA [@guilloteau_rotated_2020] in upcoming releases.


# Acknowledgements
We express our sincere thanks to the individuals who have enhanced our software through their valuable issue reports and insightful feedback.

This work forms part of the Climate Advanced Forecasting of sub-seasonal Extremes (CAFE) project, undertaken within the Physics doctoral program at the Autonomous University of Barcelona. NR acknowledges the support of the European Union’s Horizon 2020 research and innovation program, which has funded this work under the Marie Skłodowska-Curie grant (agreement No 813844).


# References