
=============================================
Handling Missing Values
=============================================

Conventional SVD algorithms aren't typically designed to manage missing values. To address this, xeofs handles missing values (``NaNs``) within your data. There are two primary types of missing values:

1. Full-dimensional: ``NaNs`` spanning all samples for a specific feature or vice versa.
2. Isolated: Occasional or sporadic ``NaNs`` within the dataset.

For example, in a 3D dataset with dimensions (time, lon, lat), a full-dimensional ``NaN`` might represent a grid point (lon, lat) exhibiting ``NaNs`` across all time steps. Conversely, an isolated ``NaN`` might indicate a grid point (lon, lat) displaying ``NaNs`` for only certain time steps.

xeofs is adept at handling full-dimensional ``NaNs``. However, it cannot manage isolated ``NaNs``. Users need to decide how to fill or remove features or samples containing isolated ``NaNs``. 

.. note::

    xeofs provides an optional runtime check ``check_nans``, enabled by default, which raises an error if isolated ``NaNs`` are detected.

