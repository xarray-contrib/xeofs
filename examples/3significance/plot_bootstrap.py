"""
Significance via bootstrap
==========================

Testing the significance of individual modes and obtain confidence intervals
for both EOFs and PCs.
"""


# Load packages and data:
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from cartopy.crs import Orthographic, PlateCarree

from xeofs.xarray import EOF, Bootstrapper

#%%

t2m = xr.tutorial.load_dataset('air_temperature')['air']

#%%
# Perform EOF analysis

model = EOF(t2m, n_modes=25, norm=False, dim='time')
model.solve()
expvar = model.explained_variance_ratio()
eofs = model.eofs()
pcs = model.pcs()


#%%
# Perform bootstrapping of the model to identy the number of significant modes.
# We choose a significance level of alpha=0.05 and perform 25 bootstraps.
# Note - if computationallly feasible - you typically want to choose higher
# numbers of bootstraps e.g. 100 or 1000.

alpha = .05
n_boot = 25

bs = Bootstrapper(n_boot=n_boot, alpha=alpha)
bs.bootstrap(model)
n_significant_modes = bs.n_significant_modes()
print('{:} modes are significant at alpha={:.2}'.format(n_significant_modes, alpha))

#%%
# The bootstrapping procedure identifies 5 significant modes. We can also
# compute the 95 % confidence intervals of the EOFs/PCs and mask out
# insignificant elements of the obtained EOFs.

eofs_ci, eofs_mask = bs.eofs()
pcs_ci, pcs_mask = bs.pcs()

#%%
# Summarize the results in a figure.


lons, lats = np.meshgrid(eofs_mask.lon.values, eofs_mask.lat.values)
proj = Orthographic(central_latitude=30, central_longitude=-80)
kwargs = {
    'cmap' : 'RdBu', 'vmin' : -.05, 'vmax': .05, 'transform': PlateCarree()
}

fig = plt.figure(figsize=(10, 16))
gs = GridSpec(5, 2)
ax1 = [fig.add_subplot(gs[i, 0], projection=proj) for i in range(5)]
ax2 = [fig.add_subplot(gs[i, 1]) for i in range(5)]

for i, (a1, a2) in enumerate(zip(ax1, ax2)):
    a1.coastlines(color='.5')
    eofs.isel(mode=i).plot(ax=a1, **kwargs)
    a1.scatter(
        lons, lats, eofs_mask.isel(mode=i).values * .5,
        color='k', alpha=.5, transform=PlateCarree()
    )
    pcs_ci.isel(mode=i, quantile=0).plot(ax=a2, color='.3', lw='.5', label='2.5%')
    pcs_ci.isel(mode=i, quantile=1).plot(ax=a2, color='.3', lw='.5', label='97.5%')
    pcs.isel(mode=i).plot(ax=a2, lw='.5', alpha=.5, label='PC')
    a2.legend(loc=2)

plt.tight_layout()
plt.savefig('bootstrap.jpg')
