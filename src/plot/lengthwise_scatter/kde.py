import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colorizer import Colorizer
from scipy.stats import gaussian_kde

from .main import DEFAULT_CELL_LENGTH_KEY, DEFAULT_POSITION_KEY, colorbar


def scatter_kde_colors(
    ax,
    foci_df,
    cmap=sns.color_palette("plasma", as_cmap=True),
    position_key=DEFAULT_POSITION_KEY,
    cell_length_key=DEFAULT_CELL_LENGTH_KEY,
):
    norm = plt.Normalize(vmin=0, vmax=4)  # type: ignore (bug)
    colorizer = Colorizer(cmap=cmap, norm=norm)

    points = foci_df[[position_key, cell_length_key]].to_numpy(dtype=float).T
    try:
        color_from_kde = gaussian_kde(points)(points)
    except (np.linalg.LinAlgError, ValueError):
        # KDE is undefined for one point or collinear data. Keep the plot
        # useful in small datasets by assigning all points the same density.
        color_from_kde = np.ones(points.shape[1])
    color_idx = np.argsort(color_from_kde)

    l_sorted = foci_df[position_key].values[color_idx]
    cell_len_sorted = foci_df[cell_length_key].values[color_idx]
    colors_from_kde = color_from_kde[color_idx]

    # x axis: position on long axis
    # y axis: cell length

    ax.scatter(
        l_sorted,
        cell_len_sorted,
        c=colors_from_kde,
        s=5,
        edgecolor="none",
        colorizer=colorizer,
    )

    cbar = colorbar(ax, cmap, norm)
    cbar.set_label("Probability density", fontsize=11, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=12)
