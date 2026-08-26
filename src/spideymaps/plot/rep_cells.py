import numpy as np

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib_scalebar.scalebar import ScaleBar
from ..map import grid_params
from .calc import get_rep, make_plot_labels
from ..make_atlas import atlas_cell_lengths_um
from src.defs import Channel
from src.spideymaps.spideymaps_v2 import SpideyAtlas


def plot_rep_cells(
    channel: Channel,
    atlas: SpideyAtlas,
    cmap=sns.color_palette("plasma", as_cmap=True),
):
    cell_lengths_um = np.asarray(atlas_cell_lengths_um(atlas))

    # Put cells into roughly four equally populated length ranges.
    quartiles = np.percentile(cell_lengths_um, [0.0, 25.0, 50.0, 75.0, 100.0])
    length_ranges = tuple(zip(quartiles[:-1], quartiles[1:]))

    pixel_sizes_um = np.array(
        [spideymap.coords["microns_per_pixel"] for spideymap in atlas.maps.values()]
    )
    if not np.allclose(pixel_sizes_um, pixel_sizes_um[0], rtol=1e-3):
        raise ValueError(
            f'all FOVs for channel "{channel.name}" must have the same pixel size'
        )

    microns_per_pixel = float(pixel_sizes_um.mean())

    plot_labels = make_plot_labels(atlas, length_ranges, microns_per_pixel)
    rep_cell_lengths, rep_cells = get_rep(
        length_ranges, cell_lengths_um, microns_per_pixel
    )

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
    axs = np.atleast_1d(axs)
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
            density = 0.0
            try:
                density = float(atlas.data[plot_label].to_dict()[key])
            except Exception as _:
                pass
            axs[i].fill(
                x,
                y,
                facecolor=sm.to_rgba(np.asarray(density)),  # scalar mappable
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
        (0.92, 0.15, 0.02, 0.7)
    )  # [left, bottom, width, height]
    cb = plt.colorbar(mappable=sm, cax=cbar_ax)
    # cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=11)
    cb.set_label("Density relative to mean", fontsize=16, rotation=270, labelpad=20)

    return fig_maps, axs
