class DimensionRenamer:
    """Rename dimensions of an xarray DataArray.

    Parameters
    ----------
    dim : str
        Name of the dimension to be renamed. Can be a dimension containing a MultiIndex.
    suffix : str
        Suffix to be added to the dimension name.

    """

    def __init__(self, dim, suffix):
        self.suffix = suffix
        self.dim = dim

    def fit(self, da):
        self.dims_mapping = {
            dim: dim + self.suffix for dim in da.coords[self.dim].coords.keys()
        }

    def transform(self, da):
        for old, new in self.dims_mapping.items():
            da = da.rename({old: new})
        return da

    def fit_transform(self, da):
        self.fit(da)
        return self.transform(da)

    def inverse_transform(self, da):
        for old, new in self.dims_mapping.items():
            da = da.rename({new: old})
        return da
