"""
Sparse PCA
========================

This example demonstrates the application of sparse PCA [1]_ to sea surface temperature data. Sparse PCA is an alternative to rotated PCA, where the components are sparse, often providing a more interpretable solution.

We replicate the analysis from the original paper [1]_, which identifies the ENSO (El Niño-Southern Oscillation) as the fourth mode, representing about 1% of the total variance. The original study focused on weekly sea surface temperatures from satellite data, whereas we use monthly data from ERSSTv5. Consequently, our results may not match exactly, but they should be quite similar.

References
----------
.. [1] Erichson, N. B. et al. Sparse Principal Component Analysis via Variable Projection. SIAM J. Appl. Math. 80, 977-1002 (2020).

"""

# Load packages and data:
import matplotlib.pyplot as plt
import xarray as xr
from cartopy.crs import EqualEarth, PlateCarree
from matplotlib.gridspec import GridSpec

import xeofs as xe

# %%
# We use sea surface temperature data from 1990 to 2017, consistent with the original paper.

sst = xr.tutorial.open_dataset("ersstv5")["sst"]
sst = sst.sel(time=slice("1990", "2017"))

# %%
# We perform sparse PCA using the `alpha` and `beta` parameters, which define the sparsity imposed by the elastic net (refer to Table 1 in the paper). In our analysis, we set `alpha` to 1e-5, as specified by the authors. Although the authors do not specify a value for `beta`, it appears that the results are not highly sensitive to this parameter. Therefore, we use the default `beta` value of 1e-4.

model = xe.single.SparsePCA(n_modes=4, alpha=1e-5)
model.fit(sst, dim="time")
expvar = model.explained_variance()
expvar_ratio = model.explained_variance_ratio()
components = model.components()
scores = model.scores()

# %%
# The explained variance fraction confirms that the fourth mode explains about 1% of the total variance, which is consistent with the original paper.

print("Explained variance: ", expvar.round(0).values)
print("Relative: ", (expvar_ratio * 100).round(1).values)

# %%
# Examining the first four modes, we clearly identify ENSO as the fourth mode.

proj = EqualEarth(central_longitude=180)
kwargs = {"cmap": "RdBu", "vmin": -0.05, "vmax": 0.05, "transform": PlateCarree()}

fig = plt.figure(figsize=(10, 12))
gs = GridSpec(4, 2, width_ratios=[1, 2])
ax0 = [fig.add_subplot(gs[i, 0]) for i in range(4)]
ax1 = [fig.add_subplot(gs[i, 1], projection=proj) for i in range(4)]

for i, (a0, a1) in enumerate(zip(ax0, ax1)):
    scores.sel(mode=i + 1).plot(ax=a0)
    a1.coastlines(color=".5")
    components.sel(mode=i + 1).plot(ax=a1, **kwargs)

    a0.set_xlabel("")

plt.tight_layout()
plt.savefig("sparse_pca.jpg")

# %%
