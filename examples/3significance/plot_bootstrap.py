"""
Significance via bootstrap
==========================

Testing the significance of individual modes and obtain confidence intervals
for both EOFs and PCs.
"""


# Load packages and data:
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
# We choose an significance level of alpha=0.05 and perform 20 bootstraps.
# Note - if computationallly feasible - you typically want to you higher
# numbers of bootstraps e.g. 100 or 1000.

alpha = .05
n_boot = 20

bs = Bootstrapper(n_boot=n_boot, alpha=alpha)
bs.bootstrap(model)
n_significant_modes = bs.n_significant_modes()
print('{:} modes are significant at alpha={:.2}'.format(n_significant_modes, alpha))

#%%
# Create figure showing the first two modes

# proj = Orthographic(central_latitude=30, central_longitude=-80)
# kwargs = {
#     'cmap' : 'RdBu', 'vmin' : -.05, 'vmax': .05, 'transform': PlateCarree()
# }
#
# fig = plt.figure(figsize=(10, 10))
# gs = GridSpec(3, 4)
# ax1 = fig.add_subplot(gs[0, :])
# ax2 = fig.add_subplot(gs[1, 2:], projection=proj)
# ax3 = fig.add_subplot(gs[1, :2])
# ax4 = fig.add_subplot(gs[2, 2:], projection=proj)
# ax5 = fig.add_subplot(gs[2, :2])
#
# ax2.coastlines(color='.5')
# ax4.coastlines(color='.5')
#
# expvar.plot(ax=ax1, marker='.')
# eofs.sel(mode=1).plot(ax=ax2, **kwargs)
# pcs.sel(mode=1).plot(ax=ax3)
# eofs.sel(mode=2).plot(ax=ax4, **kwargs)
# pcs.sel(mode=2).plot(ax=ax5)
# plt.tight_layout()
# plt.savefig('eof-smode.jpg')
