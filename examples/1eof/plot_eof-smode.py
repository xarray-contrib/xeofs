"""
EOF analysis (S-mode)
========================

EOF analysis in S-mode maximises the temporal variance.
"""


# Load packages and data:
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from cartopy.crs import EqualEarth, PlateCarree

from xeofs.xarray import EOF

#%%

sst = xr.tutorial.open_dataset('ersstv5')['sst']

#%%
# Perform the actual analysis

model = EOF(sst, n_modes=5, norm=False, dim='time')
model.solve()
expvar = model.explained_variance_ratio()
eofs = model.eofs()
pcs = model.pcs()

#%%
# Explained variance fraction
expvar * 100

#%%
# Create figure showing the first two modes

proj = EqualEarth(central_longitude=180)
kwargs = {
    'cmap' : 'RdBu', 'vmin' : -.05, 'vmax': .05, 'transform': PlateCarree()
}

fig = plt.figure(figsize=(10, 8))
gs = GridSpec(3, 2, width_ratios=[1, 2])
ax0 = [fig.add_subplot(gs[i, 0]) for i in range(3)]
ax1 = [fig.add_subplot(gs[i, 1], projection=proj) for i in range(3)]

for i, (a0, a1) in enumerate(zip(ax0, ax1)):
    pcs.sel(mode=i+1).plot(ax=a0)
    a1.coastlines(color='.5')
    eofs.sel(mode=i+1).plot(ax=a1, **kwargs)

    a0.set_xlabel('')

plt.tight_layout()
plt.savefig('eof-smode.jpg')
