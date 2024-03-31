# %%
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotx
from matplotlib.colors import LogNorm

# Load timings
# =============================================================================
timings = xr.open_dataset("timings.nc").sortby("n_features").sortby("n_samples")
xeofs_timings = (
    timings[["xeofs", "xeofs_dask"]].to_array("package").min(("run", "package"))
)
eofs_timings = (
    timings[["eofs", "eofs_dask"]].to_array("package").min(("run", "package"))
)
ratio = xeofs_timings / eofs_timings
# %%


class MidPointLogNorm(LogNorm):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        LogNorm.__init__(self, vmin=vmin, vmax=vmax, clip=clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        x, y = (
            [np.log(self.vmin), np.log(self.midpoint), np.log(self.vmax)],
            [
                0,
                0.5,
                1,
            ],
        )
        return np.ma.array(np.interp(np.log(value), x, y), mask=result.mask, copy=False)


def to_scientific_notation(arr):
    return np.array([f"$10^{len(str(x))-1}$" if x != 0 else "0" for x in arr])


# %%
sns.set_context("paper")
sns.set_style("ticks")


def create_figure(path=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ticks = [0.1, 1, 10, 60]
    ticklabels = ["100 ms", "1 s", "10 s", "1 min"]
    sns.heatmap(
        timings["xeofs_dask"].min("run"),
        ax=axes[0],
        cmap="viridis",
        norm=LogNorm(vmin=0.1, vmax=60),
        cbar_kws=dict(ticks=ticks),
    )
    axes[0].collections[0].colorbar.ax.yaxis.set_ticks([], minor=True)
    axes[0].collections[0].colorbar.set_ticklabels([t for t in ticklabels])
    xcoords = timings["xeofs"]["n_features"].values
    ycoords = timings["xeofs"]["n_samples"].values
    xticklabels = to_scientific_notation(xcoords)
    yticklabels = to_scientific_notation(ycoords)
    xticks = np.arange(0.5, len(xcoords) + 0.5, 3)
    yticks = np.arange(0.5, len(ycoords) + 0.5, 3)
    axes[0].xaxis.set_ticks(xticks)
    axes[0].yaxis.set_ticks(yticks)
    axes[0].xaxis.set_ticklabels(xticklabels[::3])
    axes[0].yaxis.set_ticklabels(yticklabels[::3], rotation=0)
    axes[0].set_xlabel("Number of features")
    axes[0].set_ylabel("Number of samples")
    axes[0].set_title("A | Runtime xeofs", loc="left")

    ticks = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    ticklabels = [
        "x1000",
        "x100",
        "x10",
        "same",
        "x10",
        "x100",
        "x1000",
    ]

    sns.heatmap(
        ratio,
        ax=axes[1],
        cmap="coolwarm",
        norm=MidPointLogNorm(vmin=0.001, vmax=1000, midpoint=1),
        cbar_kws=dict(ticks=ticks),
    )
    axes[1].collections[0].colorbar.ax.yaxis.set_ticks([], minor=True)
    axes[1].collections[0].colorbar.set_ticklabels([t for t in ticklabels])
    axes[1].collections[0].colorbar.set_label(
        "Speed\nGain", rotation=0, labelpad=40, va="center", ha="right"
    )
    xcoords = timings["xeofs"]["n_features"].values
    ycoords = timings["xeofs"]["n_samples"].values
    xticklabels = to_scientific_notation(xcoords)
    yticklabels = to_scientific_notation(ycoords)
    xticks = np.arange(0.5, len(xcoords) + 0.5, 3)
    yticks = np.arange(0.5, len(ycoords) + 0.5, 3)
    axes[1].xaxis.set_ticks(xticks)
    axes[1].yaxis.set_ticks(yticks)
    axes[1].xaxis.set_ticklabels(xticklabels[::3])
    axes[1].yaxis.set_ticklabels(yticklabels[::3], rotation=0)
    axes[1].set_xlabel("Number of features")
    axes[1].set_ylabel("Number of samples")
    axes[1].text(1.35, 1, "eofs", transform=axes[1].transAxes, ha="left", va="center")
    axes[1].text(1.35, 0, "xeofs", transform=axes[1].transAxes, ha="left", va="center")
    x = timings.n_features.values
    y = timings.n_samples.values
    xx, yy = np.meshgrid(np.arange(0.5, len(x) + 0.5), np.arange(0.5, len(y) + 0.5))
    axes[1].contour(
        xx,
        yy,
        timings["nbytes"],
        levels=[3e6],
        linestyles="--",
        colors="k",
        linewidths=0.5,
    )
    axes[1].set_title("B | Speed gain eofs vs xeofs", loc="left")
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=200)
    plt.show()


# %%
create_figure("timings_light.png")
with plt.style.context(matplotx.styles.aura["dark"]):
    create_figure("timings_dark.png")
# %%
