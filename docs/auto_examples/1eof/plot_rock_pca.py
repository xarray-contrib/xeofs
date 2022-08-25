"""
ROCK PCA
========================

Rotated complex kernel PCA by Buseo et al. (2020)

Original reference
    Bueso, D., Piles, M. & Camps-Valls, G. Nonlinear PCA for Spatio-Temporal
    Analysis of Earth Observation Data. IEEE Trans. Geosci. Remote Sensing 1–12
    (2020) doi:10.1109/TGRS.2020.2969813.
"""


# Load packages and data:
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from cartopy.crs import EqualEarth, PlateCarree

from xeofs.xarray import ROCK_PCA

#%%

sst = xr.tutorial.open_dataset('ersstv5')['sst']

#%%
# Perform the actual analysis

model = ROCK_PCA(
    sst, weights='coslat', n_modes=20, n_rot=5,
    sigma=1e4, power=2, norm=False, dim='time'
)
model.solve()
expvar = model.explained_variance_ratio()
amp = model.eofs_amplitude()
phase = model.eofs_phase()
pcs = model.pcs().real

#%%
# Create figure showing the first two modes

proj = EqualEarth(central_longitude=180)
kwargs1 = {'cmap' : 'viridis', 'transform': PlateCarree()}
kwargs2 = {'cmap' : 'twilight', 'transform': PlateCarree()}

fig = plt.figure(figsize=(14, 8))
gs = GridSpec(2, 2)
ax1 = fig.add_subplot(gs[0, 0], projection=proj)
ax2 = fig.add_subplot(gs[0, 1], projection=proj)
ax3 = fig.add_subplot(gs[1, :])

ax1.coastlines(color='.5')
ax2.coastlines(color='.5')

expvar.plot(ax=ax1, marker='.')
amp.sel(mode=1).plot(ax=ax1, **kwargs1)
phase.sel(mode=1).plot(ax=ax2, **kwargs2)
pcs.sel(mode=1).plot(ax=ax3)
plt.tight_layout()
plt.savefig('rock-pca.jpg')
