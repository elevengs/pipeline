import numpy as np

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import seaborn as sns


from matplotlib_scalebar.scalebar import ScaleBar
grid_params = dict(
    radius=6,
    n_shells=5,
    n_cols=5,
    n_phi=(1, 3, 5, 5, 5),
    level=0,
)

def plot_spideymaps(
    atlas,
    rep_cells,
    rep_cell_lengths,
    plot_labels,
    microns_per_pixel,
    cmap=sns.color_palette("plasma", as_cmap=True),
):
    vmin = 0
    vmax = 3

    max_length = rep_cell_lengths.max()
    xlim = (-max_length / 2 * 1.1, max_length / 2 * 1.1)
    ylim = (
        -grid_params["radius"] * 1.1 * microns_per_pixel,
        grid_params["radius"] * 1.1 * microns_per_pixel,
    )

    fig_maps, axs = plt.subplots(
        len(plot_labels), 1, figsize=(7, 7 * len(plot_labels) * (ylim[1] / xlim[1]))
    )
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax, clip=True), cmap=cmap)

    for i, (rep_cell, plot_label) in enumerate(zip(rep_cells, plot_labels)):
        for key in rep_cell.polygons:
            x = (
                np.array(rep_cell.polygons[key].boundary.xy[0], dtype="float")
                * microns_per_pixel
            )
            y = (
                np.array(rep_cell.polygons[key].boundary.xy[1], dtype="float")
                * microns_per_pixel
            )
            density = 0
            try:
                density = atlas.data[plot_label].to_dict()[key]
            except Exception as _:
                pass
            axs[i].fill(
                x,
                y,
                facecolor=sm.to_rgba(density),  # scalar mappable
                edgecolor="none",
                linewidth=0,
            )

        axs[i].set_aspect("equal")

        axs[i].spines["bottom"].set_color(4 * [0])
        axs[i].spines["top"].set_color(4 * [0])
        axs[i].spines["left"].set_color(4 * [0])
        axs[i].spines["right"].set_color(4 * [0])

        axs[i].set_xticks([])
        axs[i].set_yticks([])

        axs[i].set_xlim(xlim)
        axs[i].set_ylim(ylim)

        axs[i].set_aspect("equal")

    # Add scalebar to first subplot only
    scalebar = ScaleBar(
        1,
        "um",
        color="black",
        box_alpha=0,
        width_fraction=0.05,
        location="upper left",
        font_properties={"size": 12},
    )
    axs[0].add_artist(scalebar)

    # Create a new axes for the colorbar
    cbar_ax = fig_maps.add_axes(
        [0.92, 0.15, 0.02, 0.7]
    )  # [left, bottom, width, height]
    cb = plt.colorbar(mappable=sm, cax=cbar_ax)
    # cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=11)
    cb.set_label("Density relative to mean", fontsize=16, rotation=270, labelpad=20)

    return fig_maps, axs
