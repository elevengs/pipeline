import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

DEFAULT_POSITION_KEY = "FOCUS::MIDLINE_POSITION_POL_UM"
DEFAULT_CELL_LENGTH_KEY = "CELL::MIDLINE_LENGTH_UM"


def colorbar(ax, cmap, norm):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    return plt.colorbar(sm, cax=cax)


def plot_lengthwise_scatter(
    foci_df_seq,
    min_plot_len=0.8,
    max_plot_len=2.1,
    scatter_fn=None,
    cmap=sns.color_palette("plasma", as_cmap=True),
    position_key=DEFAULT_POSITION_KEY,
    cell_length_key=DEFAULT_CELL_LENGTH_KEY,
):
    if scatter_fn is None:
        from .kde import scatter_kde_colors

        scatter_fn = scatter_kde_colors

    all = pd.concat(foci_df_seq)

    all = all[np.isfinite(all[position_key])]
    all = all[np.isfinite(all[cell_length_key])]

    fig_cano, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=600)

    scatter_fn(
        ax,
        all,
        cmap=cmap,
        position_key=position_key,
        cell_length_key=cell_length_key,
    )

    # plot min and max bars
    # right bar: (min cell len / 2, min cell len) to (max cell len / 2, max cell len)
    buf = 0.025
    ax.plot(
        [min_plot_len / 2 + buf, max_plot_len / 2 + buf],
        [min_plot_len, max_plot_len],
        "k-",
    )
    ax.plot(
        [-min_plot_len / 2 - buf, -max_plot_len / 2 - buf],
        [min_plot_len, max_plot_len],
        "k-",
    )

    ax.plot([0, 0], [min_plot_len, max_plot_len], "k:")

    ax.invert_yaxis()

    ax.set_aspect("equal")

    ax.set_xlim(-max_plot_len / 2, max_plot_len / 2)
    ax.set_ylim(max_plot_len, min_plot_len)

    # Hide axes and borders
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_xlabel("Long axis position", fontsize=12)
    ax.set_ylabel(f"Cell length (n = {len(all)})", fontsize=12)

    # Add scalebar
    scalebar = ScaleBar(
        1,
        "um",
        color="black",
        box_alpha=0,
        width_fraction=0.02,
        location="upper left",
        font_properties={"size": 10},
    )
    ax.add_artist(scalebar)

    return fig_cano
