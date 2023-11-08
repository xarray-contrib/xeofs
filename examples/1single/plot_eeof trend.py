"""
Removing nonlinear trends with EEOF analysis
============================================

This example demonstrates how to use Extended EOF (EEOF) analysis 
in order to remove nonlinear trends from a dataset.

Let's begin by setting up the required packages and fetching the data:
"""

import xarray as xr
import xeofs as xe
import matplotlib.pyplot as plt

xr.set_options(display_expand_data=False)

# %%
# Load the tutorial data.

sst = xr.tutorial.open_dataset("ersstv5").sst
sst = sst.groupby("time.month") - sst.groupby("time.month").mean("time")


# %%
# Prior to conducting the EEOF analysis, it's essential to determine the
# structure of the lagged covariance matrix. This entails defining the time
# delay ``tau`` and the ``embedding`` dimension. The former signifies the
# interval between the original and lagged time series, while the latter
# dictates the number of time-lagged copies in the delay-coordinate space,
# representing the system's dynamics.
# For illustration, using ``tau=4`` and ``embedding=40``, we generate 40
# delayed versions of the time series, each offset by 4 time steps, resulting
# in a maximum shift of ``tau x embedding = 160``. Given our dataset's
# 6-hour intervals, tau = 4 translates to a 24-hour shift.
# It's obvious that this way of constructing the lagged covariance matrix
# and subsequently decomposing it can be computationally expensive. For example,
# given our dataset's dimensions,

eof = xe.models.EOF(n_modes=10)
eof.fit(sst, dim="time")
scores = eof.scores()
components = eof.components()

# %%

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
scores.sel(mode=1).plot(ax=ax[0])
components.sel(mode=1).plot(ax=ax[1])
plt.show()

# %%
# EEOF analysis

eeof = xe.models.ExtendedEOF(n_modes=5, tau=1, embedding=120, n_pca_modes=50)
eeof.fit(sst, dim="time")
components_ext = eeof.components()
scores_ext = eeof.scores()

# %%

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
scores_ext.sel(mode=1).plot(ax=ax[0])
components_ext.sel(mode=1, embedding=0).plot(ax=ax[1])
plt.show()

# %%
# Remov the trend

sst_trends = eeof.inverse_transform(scores_ext.sel(mode=1))
sst_detrended = sst - sst_trends.drop_vars("mode")


# %%
# EOF analysis on detrended data

eof_model_detrended = xe.models.EOF(n_modes=5)
eof_model_detrended.fit(sst_detrended, dim="time")
scores_detrended = eof_model_detrended.scores()
components_detrended = eof_model_detrended.components()


# %%

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
scores_detrended.sel(mode=1).plot(ax=ax[0])
components_detrended.sel(mode=1, embedding=0).plot(ax=ax[1])
plt.show()
