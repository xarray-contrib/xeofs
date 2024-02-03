![xeofs logo](docs/logos/xeofs_logo.png)

| Versions                   | ![PyPI](https://img.shields.io/pypi/v/xeofs) ![Conda](https://img.shields.io/conda/vn/conda-forge/xeofs) |
|----------------------------|:---------------------------------------------------------------------------------------------:|
| Build & Testing            | ![Build](https://img.shields.io/github/actions/workflow/status/xarray-contrib/xeofs/ci.yml?branch=main) ![Coverage](https://codecov.io/gh/xarray-contrib/xeofs/branch/main/graph/badge.svg?token=8040ZDH6U7) |
| Code Quality               | ![Black](https://img.shields.io/badge/code%20style-black-000000.svg)                           |
| Documentation              | ![Docs](https://readthedocs.org/projects/xeofs/badge/?version=latest)                          |
| Citation & Licensing       | [![JOSS](https://joss.theoj.org/papers/4f50349ee1777b8a61761183047b1180/status.svg)](https://joss.theoj.org/papers/4f50349ee1777b8a61761183047b1180) ![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.6323012.svg) ![License](https://img.shields.io/pypi/l/xeofs) |
| User Engagement            | ![Downloads](https://img.shields.io/pypi/dw/xeofs)                                          |

## Overview

`xeofs` is a dedicated Python package for dimensionality reduction in the realm of climate science, offering methods like PCA, known as EOF analysis within the field, and related variants. Seamlessly integrated with `xarray` and `Dask`, it's tailored for easy handling and scalable computation on large, multi-dimensional datasets, making advanced climate data analysis both accessible and efficient.

- **Multi-Dimensional**: Designed for `xarray` objects, it applies dimensionality reduction to multi-dimensional data while maintaining data labels.
- **Dask-Integrated**: Supports large datasets via `Dask` xarray objects
- **Extensive Methods**: Offers various dimensionality reduction techniques
- **Adaptable Output**: Provides output corresponding to the type of input, whether single or list of `xr.DataArray` or `xr.Dataset`
- **Missing Values**: Handles `NaN` values within the data
- **Bootstrapping**: Comes with a user-friendly interface for model evaluation using bootstrapping
- **Efficient**: Ensures computational efficiency, particularly with large datasets through randomized SVD
- **Modular**: Allows users to implement and incorporate new dimensionality reduction methods

## Installation

To install the package, use either of the following commands:

```bash
conda install -c conda-forge xeofs
```

or

```bash
pip install xeofs
```

## Quickstart

In order to get started with `xeofs`, follow these simple steps:

**Import the package**
    
```python
>>> import xarray as xr  # for example data only
>>> import xeofs as xe

```

**Load example data**

```python
>>> t2m = xr.tutorial.open_dataset("air_temperature")
>>> t2m_west = t2m.isel(lon=slice(None, 20))
>>> t2m_east = t2m.isel(lon=slice(21, None))

```

**EOF analysis**
Initiate and fit the EOF/PCA model to the data

```python
>>> eof = xe.models.EOF(n_modes=10)
>>> eof.fit(t2m, dim="time")  # doctest: +ELLIPSIS
<xeofs.models.eof.EOF object at ...>

```
Now, you can access the model's EOF components and PC scores:

```py
>>> comps = eof.components()  # EOFs (spatial patterns)
>>> scores = eof.scores()  # PCs (temporal patterns)

```

**Varimax-rotated EOF analysis**
Initiate and fit an `EOFRotator` class to the model to obtain a varimax-rotated EOF analysis

```python
>>> rotator = xe.models.EOFRotator(n_modes=3)
>>> rotator.fit(eof) # doctest: +ELLIPSIS
<xeofs.models.eof_rotator.EOFRotator object at ...>

>>> rot_comps = rotator.components()  # Rotated EOFs (spatial patterns)
>>> rot_scores = rotator.scores()  # Rotated PCs (temporal patterns)

```

**Maximum Covariance Analysis (MCA)**

```python
>>> mca = xe.models.MCA(n_modes=10)
>>> mca.fit(t2m_west, t2m_east, dim="time")  # doctest: +ELLIPSIS
<xeofs.models.mca.MCA object at ...>

>>> comps1, comps2 = mca.components()  # Singular vectors (spatial patterns)
>>> scores1, scores2 = mca.scores()  # Expansion coefficients (temporal patterns)

```

**Varimax-rotated MCA**

```python
>>> rotator = xe.models.MCARotator(n_modes=10)
>>> rotator.fit(mca)  # doctest: +ELLIPSIS
<xeofs.models.mca_rotator.MCARotator object at ...>

>>> rot_comps = rotator.components()  # Rotated singular vectors (spatial patterns)
>>> rot_scores = rotator.scores()  # Rotated expansion coefficients (temporal patterns)

```

To further explore the capabilities of `xeofs`, check out the [available documentation](https://xeofs.readthedocs.io/en/latest/) and [examples](https://xeofs.readthedocs.io/en/latest/auto_examples/index.html).
For a full list of currently available methods, see the [Reference API](https://xeofs.readthedocs.io/en/latest/api.html).

## Documentation

For a more comprehensive overview and usage examples, visit the [documentation](https://xeofs.readthedocs.io/en/latest/).

## Contributing

Contributions are highly welcomed and appreciated. If you're interested in improving `xeofs` or fixing issues, please read our [Contributing Guide](https://xeofs.readthedocs.io/en/latest/overview_3_contributing.html).

## License

This project is licensed under the terms of the [MIT license](https://github.com/xarray-contrib/xeofs/blob/main/LICENSE).

## Contact

For questions or support, please open a [Github issue](https://github.com/xarray-contrib/xeofs/issues).

## Credits

- Randomized PCA: [scikit-learn](https://scikit-learn.org/stable/)
- EOF analysis: Python package [eofs](https://github.com/ajdawson/eofs) by Andrew Dawson
- MCA: Python package [xMCA](https://github.com/Yefee/xMCA) by Yefee
- CCA: Python package [CCA-Zoo](https://github.com/jameschapman19/cca_zoo) by James Chapman
- ROCK-PCA: Matlab implementation by [Diego Bueso](https://github.com/DiegoBueso/ROCK-PCA)

## How to cite?

When using `xeofs`, kindly remember to cite the original references of the methods employed in your work. Additionally, if `xeofs` is proving useful in your research, we'd appreciate if you could acknowledge its use with the following citation:

```bibtex
@article{rieger_xeofs_2024,
author = {Rieger, Niclas and Levang, Samuel J.},
doi = {10.21105/joss.06060},
journal = {Journal of Open Source Software},
month = jan,
number = {93},
pages = {6060},
title = {{xeofs: Comprehensive EOF analysis in Python with xarray}},
url = {https://joss.theoj.org/papers/10.21105/joss.06060},
volume = {9},
year = {2024}
}
```

## Contributors

[![Contributors](https://contrib.rocks/image?repo=xarray-contrib/xeofs)](https://github.com/xarray-contrib/xeofs/graphs/contributors)
