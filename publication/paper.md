
# Co-publication of science, methods, and software

Sometimes authors prepare a JOSS publication alongside a contribution describing a science application, details of algorithm development, and/or methods assessment. In this circumstance, JOSS considers submissions for which the implementation of the software itself reflects a substantial scientific effort. This may be represented by the design of the software, the implementation of the algorithms, creation of tutorials, or any other aspect of the software. We ask that authors indicate whether related publications (published, in review, or nearing submission) exist as part of submitting to JOSS.

-> complex rotated MCA



# Meta data file

generate a [meta data file](https://gist.github.com/arfon/478b2ed49e11f984d6fb) and add it to the repo

# Paper

- between 250 - 1000 words
- Begin your paper with a summary of the high-level functionality of your software for a non-specialist reader. Avoid jargon in this section.
  
it should include:

- [] A list of the authors of the software and their affiliations, using the correct format (see the example below).
- [] A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience.
- [] A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work.
- [] A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline.
- [] Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it.
- [] Acknowledgement of any financial support.


- [] Review [checklist](https://joss.readthedocs.io/en/latest/review_checklist.html)
- [] Review [critera](https://joss.readthedocs.io/en/latest/review_criteria.html)


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
    affiliation: "1, 2"
  - name: Samuel J. Levang
    affiliation: 3
  - name: Aaron Spring
    affiliation: 4
affiliations:
  - name: Centre de Recerca Matemàtica (CRM), Bellaterra, Spain
    index: 1
  - name: UAB
    index: 1
  - name: Instituto de Ciencias del Mar (ICM) - CSIC, Barcelona, Spain
    index: 2
  - name: TBF
    index: 3
  - name: TBF
    index: 4
date: 30 October 2023
bibliography: paper.bib

---

# Summary

`xeofs` is a Python package designed specifically for the climate science 
community to facilitate advanced data analysis through dimensionality 
reduction techniques such as Empirical Orthogonal Functions (EOF) analysis, 
commonly referred to as Principal Component Analysis (PCA) in other fields. 
Tailored to work natively with xarray objects [@hoyer2017xarray], xeofs simplifies the analysis 
of large, labeled, multi-dimensional datasets. Leveraging the power of Dask [@dask_2016], 
it allows users to scale their computations efficiently across cores or 
even clusters, making it suitable for high-volume climate data applications.


# Statement of need

Climate science typically involves the analysis of multi-dimensional, large-scale datasets.
The sheer volume of data, coupled with the complexity of the underlying
physical processes, makes it challenging to extract meaningful information
from climate data. Historically, this fact led climate researchers to seek for
and develop a wide range of dimensionality reduction techniques, many of which
are variants of Empirical Orthogonal Functions (EOF) analysis, commonly referred
to as Principal Component Analysis (PCA) in other fields. While [@pedregosa_scikit-learn_2011]
offers a comprehensive collection of the most commonly used dimensionality reduction
and pattern recognition techniques, climate scientists often require more
specialized tools to handle the unique challenges posed by climate data, ranging from
established PCA variants [@hannachi_empirical_2007] to more recent developments [@@hannachi_patterns_2021] including 
ROCK-PCA [@bueso_nonlinear_2020] and spectral, rotated PCA [@guilloteau_rotated_2020].

Additionally, most of these dimensionality reduction techniques require data to be
presented in a 2D matrix representing *samples* and *features*. 
However, climate data is often multi-dimensional, most commonly represented by 
but not restricted 
to time, longitude and latitude dimensions requiring careful
transformations to be applied prior to and after the analysis to recover the
original dimensions. Due to the nature of climate data, data sets are often filled 
with missing values representing ocean or land that the user may want to recover after 
the analysis thus requiring tracking and propagation of missing values. 
These pre- and postprocessing steps are typically time-intensive and error-prone, 
which can quickly lead to a large overhead in the analysis time, in particular for 
smaller projects.
Moreover, climate data are often large, requiring out-of-memory processing
techniques to be applied. Over the recent years, the Python ecosystem has
witnessed the emergence of a number of packages that aim to address these
challenges, including `xarray` [@hoyer2017xarray] and `dask` [@dask_2016],
providing both a framework to handle labeled, multi-dimensional datasets and
a library for parallel computing, respectively. 

Python packages like `xMCA` or `eofs` [@dawson2016eofs] have tapped into this void,
the latter by providing a user-friendly implementation for PCA/EOF analysis using 
`xarray` and `Dask`. With `xeofs` we go beyond these capabilities 
by providing a more comprehensive toolbox for dimensionality reduction techniques, 
including methods like rotated [], complex/Hilbert and extended [] EOF analysis. 
Moreover, `xeofs` is designed to work natively with `xarray` objects, thereby retaining
data labels and structure, and seamlessly integrates with `dask` for handling
out-of-memory datasets. Due to its usage of randomized Singular Value Decomposition (SVD)
`xeofs` also computationlly outperforms for larger data sets. 
Further, it eliminates the need to flatten data into 2D matrices,
preserving coordinates and simplifying subsequent analysis. With `xeofs`, users can
perform analyses on datasets containing missing values which are automatically
masked during and recovered after the analysis. Furthermore, it provides users
with an easy-to-use bootstrapping module to evalute model results. Moreover,
`xeofs` offers a user-friendly interface to provide users with a low-level entry point.
Finally, `xeofs` is designed to be extensible, allowing users to easily add their
own dimensionality reduction techniques therby exploiting the internal pre and post 
processing pipeline of `xeofs`.


| Method                        | Alternative name                                                | Reference       |
| :---                          | :---                                                            |     :---:       |
| PCA                           | EOF analysis                                                    | git status      |
| Promax-rotated PCA            |                                                                 | git status      |
| Complex PCA                   | Hilbert EOF (HEOF) analysis                                     |                 |
| Complex rotated PCA           | -                                                               |                 |
| Extended PCA                  | EEOF analysis / Multichannel Singular Spectrum Analysis (M-SSA) |                 |
| Optimal Persistence Analysis  | OPA                                                             |                 |
| Geographically-weighted PCA   | GWPCA                                                           |                 |
| Maximum Covariance Analysis   | MCA                                                             |                 |
| Complex MCA                   | Hilbert MCA/Analytic MCA                                        |                 |
| Canonical Correlation Analysis| CCA                                                             |                 |



# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }



# Acknowledgements

--> UAB sentence and CAFE funding
-- all people who contributed to the software by raising issues.


# References