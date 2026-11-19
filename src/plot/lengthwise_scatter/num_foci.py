from src.plot.config import DEFAULT_PLOT_CONFIG

from matplotlib import pyplot as plt
from matplotlib.colorizer import Colorizer
from matplotlib.colors import BoundaryNorm

from .main import DEFAULT_CELL_LENGTH_KEY, DEFAULT_POSITION_KEY, colorbar


def scatter_num_foci(
    ax,
    foci_df,
    cmap=plt.colormaps["okabe_ito"],
    position_key=DEFAULT_POSITION_KEY,
    cell_length_key=DEFAULT_CELL_LENGTH_KEY,
    colorbar_min=DEFAULT_PLOT_CONFIG.lengthwise_scatter_num_foci_colorbar_min,
    colorbar_max=DEFAULT_PLOT_CONFIG.lengthwise_scatter_num_foci_colorbar_max,
):
    if colorbar_min == 1 and colorbar_max is None:
        norm = BoundaryNorm([1, 2, 3, cmap.N], cmap.N, extend="both")
    else:
        upper = cmap.N if colorbar_max is None else colorbar_max
        boundaries = [count - 0.5 for count in range(colorbar_min, upper + 2)]
        n_colors = len(boundaries) + 1
        cmap = cmap.resampled(max(cmap.N, n_colors))
        norm = BoundaryNorm(boundaries, cmap.N, extend="both")
    colorizer = Colorizer(cmap=cmap, norm=norm)

    l_sorted = foci_df[position_key]
    cell_len_sorted = foci_df[cell_length_key]

    color = foci_df["CELL_LAYER::PROPS::NUM_FOCI"]

    ax.scatter(
        l_sorted,
        cell_len_sorted,
        c=color,
        s=4,
        edgecolor="none",
        alpha=1,
        colorizer=colorizer,
    )

    cbar = colorbar(ax, cmap, norm)
    cbar.set_label("Number of foci", fontsize=11, rotation=270, labelpad=20)
