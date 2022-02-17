"""
EOF analysis (T-mode)
-----------------------

EOF analysis in T-mode maximises the spatial variance.

Load packages and data:
"""
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from cartopy.crs import Orthographic, PlateCarree

from xeofs.xarray import EOF

t2m = xr.tutorial.load_dataset('air_temperature')['air']

#%%
# Perform the actual analysis

model = EOF(t2m, n_modes=5, norm=False, dim=['lat', 'lon'])
model.solve()
expvar = model.explained_variance_ratio()
eofs = model.eofs()
pcs = model.pcs()

#%%
# Create figure showing the first two modes

proj = Orthographic(central_latitude=30, central_longitude=-80)
kwargs = {
    'cmap' : 'RdBu', 'transform': PlateCarree()
}

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(3, 4)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :2])
ax3 = fig.add_subplot(gs[1, 2:], projection=proj)
ax4 = fig.add_subplot(gs[2, :2])
ax5 = fig.add_subplot(gs[2, 2:], projection=proj)

ax3.coastlines(color='.5')
ax5.coastlines(color='.5')

expvar.plot(ax=ax1, marker='.')
eofs.sel(mode=1).plot(ax=ax2)
pcs.sel(mode=1).plot(ax=ax3, **kwargs)
eofs.sel(mode=2).plot(ax=ax4)
pcs.sel(mode=2).plot(ax=ax5, **kwargs)
plt.tight_layout()
plt.savefig('eof-tmode.jpg')
