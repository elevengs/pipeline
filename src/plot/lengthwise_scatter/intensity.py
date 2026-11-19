from src.plot.config import DEFAULT_PLOT_CONFIG

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colorizer import Colorizer
import seaborn as sns

from .main import DEFAULT_CELL_LENGTH_KEY, DEFAULT_POSITION_KEY, colorbar


def scatter_intensity(
    ax,
    foci_df,
    cmap=sns.color_palette("plasma", as_cmap=True),
    lower_quantile_norm=0.01,
    upper_quantile_norm=0.99,
    position_key=DEFAULT_POSITION_KEY,
    cell_length_key=DEFAULT_CELL_LENGTH_KEY,
    colorbar_min=DEFAULT_PLOT_CONFIG.lengthwise_scatter_intensity_colorbar_min,
    colorbar_max=DEFAULT_PLOT_CONFIG.lengthwise_scatter_intensity_colorbar_max,
):
    l_sorted = foci_df[position_key]
    cell_len_sorted = foci_df[cell_length_key]

    color = foci_df["FOCUS::PROPS::MAX_INTENSITY"]
    lims = np.quantile(color.to_numpy(), [lower_quantile_norm, upper_quantile_norm])

    norm = plt.Normalize(
        vmin=lims[0] if colorbar_min is None else colorbar_min,
        vmax=lims[1] if colorbar_max is None else colorbar_max,
    )
    colorizer = Colorizer(cmap=cmap, norm=norm)

    ax.scatter(
        l_sorted,
        cell_len_sorted,
        c=color,
        s=2,
        edgecolor="none",
        alpha=1,
        colorizer=colorizer,
    )

    cbar = colorbar(ax, cmap, norm)
    cbar.set_label("Focus intensity", fontsize=11, rotation=270, labelpad=20)
