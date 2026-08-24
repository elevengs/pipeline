import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colorizer import Colorizer
import seaborn as sns

from .main import DEFAULT_CELL_LENGTH_KEY, DEFAULT_POSITION_KEY, colorbar


def scatter_intensity(
    ax,
    foci_df,
    cmap = sns.color_palette("plasma", as_cmap=True),
    lower_quantile_norm = 0.01,
    upper_quantile_norm = 0.99,
    position_key=DEFAULT_POSITION_KEY,
    cell_length_key=DEFAULT_CELL_LENGTH_KEY,
):
    l_sorted = foci_df[position_key]
    cell_len_sorted = foci_df[cell_length_key]

    color = foci_df["FOCUS::PROPS::MAX_INTENSITY"]
    lims = np.quantile(
        color.to_numpy(),
        [
            lower_quantile_norm,
            upper_quantile_norm
        ]
    )
    
    norm = plt.Normalize(vmin=lims[0], vmax=lims[1])
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
