"""
Significance testing of EOF analysis via bootstrap
===================================================

Test the significance of individual modes and obtain confidence intervals
for both EOFs and PCs.
"""

# Load packages and data:
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.crs import Orthographic, PlateCarree
from matplotlib.gridspec import GridSpec

import xeofs as xe

# %%

t2m = xr.tutorial.load_dataset("air_temperature")["air"]

# %%
# Perform EOF analysis

model = xe.single.EOF(n_modes=5, standardize=False)
model.fit(t2m, dim="time")
expvar = model.explained_variance_ratio()
components = model.components()
scores = model.scores()


# %%
# Perform bootstrapping of the model to identy the number of significant modes.
# We perform 50 bootstraps.
# Note - if computationallly feasible - you typically want to choose higher
# numbers of bootstraps e.g. 1000.

n_boot = 50

bs = xe.validation.EOFBootstrapper(n_bootstraps=n_boot)
bs.fit(model)
bs_expvar = bs.explained_variance()
ci_expvar = bs_expvar.quantile([0.025, 0.975], "n")  # 95% confidence intervals

q025 = ci_expvar.sel(quantile=0.025)
q975 = ci_expvar.sel(quantile=0.975)

is_significant = q025 - q975.shift({"mode": -1}) > 0
n_significant_modes = (
    is_significant.where(is_significant is True).cumsum(skipna=False).max().fillna(0)
)
print("{:} modes are significant at alpha=0.05".format(n_significant_modes.values))

# %%
# The bootstrapping procedure identifies 3 significant modes. We can also
# compute the 95 % confidence intervals of the EOFs/PCs and mask out
# insignificant elements of the obtained EOFs.

ci_components = bs.components().quantile([0.025, 0.975], "n")
ci_scores = bs.scores().quantile([0.025, 0.975], "n")

is_sig_comps = np.sign(ci_components).prod("quantile") > 0


# %%
# Summarize the results in a figure.


lons, lats = np.meshgrid(is_sig_comps.lon.values, is_sig_comps.lat.values)
proj = Orthographic(central_latitude=30, central_longitude=-80)
kwargs = {"cmap": "RdBu", "vmin": -0.05, "vmax": 0.05, "transform": PlateCarree()}

fig = plt.figure(figsize=(10, 16))
gs = GridSpec(5, 2)
ax1 = [fig.add_subplot(gs[i, 0], projection=proj) for i in range(5)]
ax2 = [fig.add_subplot(gs[i, 1]) for i in range(5)]

for i, (a1, a2) in enumerate(zip(ax1, ax2)):
    a1.coastlines(color=".5")
    components.isel(mode=i).plot(ax=a1, **kwargs)
    a1.scatter(
        lons,
        lats,
        is_sig_comps.isel(mode=i).values * 0.5,
        color="k",
        alpha=0.5,
        transform=PlateCarree(),
    )
    ci_scores.isel(mode=i, quantile=0).plot(ax=a2, color=".3", lw=".5", label="2.5%")
    ci_scores.isel(mode=i, quantile=1).plot(ax=a2, color=".3", lw=".5", label="97.5%")
    scores.isel(mode=i).plot(ax=a2, lw=".5", alpha=0.5, label="PC")
    a2.legend(loc=2)

plt.tight_layout()
plt.savefig("bootstrap.jpg")
