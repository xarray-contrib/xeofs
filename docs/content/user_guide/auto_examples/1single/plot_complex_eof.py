"""
Complex EOF analysis
============================================

In this tutorial, we'll walk through how to perform a Complex EOF analysis on
the zonal and meridional wind components.

Let's start by importing the necessary packages and loading the data:
"""

# %%
import matplotlib.pyplot as plt
import xarray as xr

import xeofs as xe

xr.set_options(display_expand_attrs=False)

# %%
# For this example, we'll use the ERA-Interim tutorial dataset ``eraint_uvz``:

uvz = xr.tutorial.open_dataset("eraint_uvz")
uvz

# %%
# This dataset contains the zonal, meridional, and vertical wind components at
# three different atmospheric levels. Note that the data only covers two months,
# so we have just two time steps (samples). While this isn't enough for a robust
# EOF analysis, we'll proceed for demonstration purposes. Now, let's combine the
# zonal (``u``) and meridional (``v``) wind components into a complex-valued
# dataset:

Z = uvz["u"] + 1j * uvz["v"]

# %%
# Next, we'll initialize and fit the ``ComplexEOF`` model to our data. The
# ``xeofs`` package makes this easy by allowing us to specify the sample
# dimension (``month``), automatically performing the Complex EOF analysis
# across all three atmospheric levels. As a standard practice, we'll also weigh
# each grid cell by the square root of the cosine of the latitude
# (``use_coslat=True``).

model = xe.single.ComplexEOF(n_modes=1, use_coslat=True, random_state=7)
model.fit(Z, dim="month")

# %%
# Instead of just extracting the complex-valued components, we can also get the
# amplitude and phase of these components. Let's start by looking at the
# amplitude of the first mode:


spatial_ampltiudes = model.components_amplitude()
spatial_phases = model.components_phase()

spatial_ampltiudes.sel(mode=1).plot(col="level")
plt.show()

# %%
# It looks like the first mode picks up a pattern resembling the location of the
# subtropical jet stream around ±30º latitude, particularly strong in the upper
# troposphere at 200 hPa and weaker toward the surface. We can also plot the
# phase of the first mode. To keep the plot clear, we'll only show the phase
# where the amplitude is above a certain threshold (e.g., 0.004):

relevant_phases = spatial_phases.where(spatial_ampltiudes > 0.004)
relevant_phases.sel(mode=1).plot(col="level", cmap="twilight")
plt.show()

# %%
