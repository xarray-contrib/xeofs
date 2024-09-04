# EOF analysis

Empirical Orthogonal Function (EOF) analysis, more known as Principal Component Analysis (PCA), has a long history in climate science since having popularied by [AUTHORS] et al in [YEAR]. [list some of its benefits and why it is so useful.]


### How does it work?

[Describe in one paragraph what EOF analysis does mathematically. Specifically, explain what the components (EOFs), the scores (PCs) and the eigenvalues (explained variance) mean.]


### An example: North American air temperatures between 2013-2014
In the following we perform EOF analysis of 2-year temperature records over the North American continent using ``xeofs``. The following cell loads all necessary packages, retrieves the data and performs the analysis.


```python
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.feature as cfeature
import ipywidgets as widgets
from IPython.display import display
from cartopy.crs import LambertAzimuthalEqualArea, PlateCarree
from xeofs.single import EOF

t2m = xr.tutorial.load_dataset('air_temperature')['air']

model = EOF(n_modes=20, standardize=True, use_coslat=True)
model.fit(t2m, dim='time')

expvar = model.explained_variance_ratio()
compontents = model.components()
scores = model.scores()
```

Let's create a plot visualising the results of EOF analysis.


```python
def plot_pca_results(mode):
    proj = LambertAzimuthalEqualArea(central_latitude=50, central_longitude=-90)
    proj_data = PlateCarree()

    fig = plt.figure(figsize=(9, 3))
    gs = GridSpec(1, 2, width_ratios=[1, 2], wspace=0.05)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], projection=proj)

    scores.sel(mode=mode).plot(ax=ax1)

    ax2.add_feature(cfeature.LAND, facecolor='.9')
    ax2.add_feature(cfeature.OCEAN, facecolor='.9')
    ax2.coastlines(color='.5', lw=0.5)

    kwargs = dict(vmin=-0.15, vmax=0.15, cmap='RdBu_r', transform=proj_data)
    compontents.sel(mode=mode).plot(ax=ax2, **kwargs)

    ax1.set_title('Explained variance: {:.1f}%'.format(expvar.sel(mode=mode).values * 100))
    ax2.set_title('')

    plt.show()

mode_slider = widgets.IntSlider(
    value=1,
    min=1,
    max=10,
    step=1,
    description='Mode:',
    continuous_update=False
)

widgets.interactive(plot_pca_results, mode=mode_slider)
```




    interactive(children=(IntSlider(value=1, continuous_update=False, description='Mode:', max=10, min=1), Output(…



### Challenge: interpretability
There are good reasons why EOF analysis is used so often in data compression taks e.g. for images. Its mathematical property of producing components which are orthogonal are one of the reasons why it is such an efficient technique in compressing information. 

In climate science, we often deal with huge data sets which would be too cumbersome to analyze time series by time series and which also inlcude often a high degree of redundancy. Therefore, compressing data sets in their most important modes of variability can help scientist to better understand the underlying patterns, dynamics and variability within the climate system. One of the prime examples of this is the El Nino Southern Oscillation (ENSO) and its associated teleconnections, where EOF analysis has played a crucial role in defining and shaping our first understanding back in [YEAR]. Until today, some of the indices representing the state of ENSO (and other well known climate patterns) are based on EOF analysis. 

However, while climate scientist are mostly concernced with understanding the the data, the EOF analysis' main goal is to compress the data as efficient as possible. One result of that is that the patterns obey the orthogonal property. While handy mathematically, there is no reason to believe that the components (apart from the first mode) bear much meaning. At the end, they are mathematical artifacts which condense the data most efficiently. This has two important implications from that:

1. Efficiently compressing means that the resulting components are **dense matrices**, i.e. the individual entries are mostly non-zero. This can make interpretation challenging because every pattern is based on the entire field.
2. When applied to climate data, the orthogonal constrain often leads to a well known spatial pattern, known as the **Buell patterns** ([AUTHOR] et al. [YEAR]). These Buell patterns often consist of a unipolar first component, a bipolar second and third component, then tripoles, quadrupoles etc. All too often these patterns are interpreted as a South-North dipole or similar, although the chances are much higher that they represent a mere mathematical artifact. 

You can see both implications play out in the example above. Just scroll through the different modes and you can see that the individual modes are all dense with almost no zero values. You can also clearly spot the Buell patterns in these modes, with mode 1 being unipolar and mode 2 and 3 being bipolar. One could argue that the first mode shows a feature important in the entire domain (e.g. the seasonal cycle) and the second mode represent the difference between the continental and oceanic parts in the seasonal cycle. But wouldn't it be simpler to just have two modes which represent the continental and oceanic seasonal cycle serparately ?

And indeed we can come closer to such a solution by applying a technique that is called Varimax/Promax rotation. This is explained the following section. 






