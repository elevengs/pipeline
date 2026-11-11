from collections.abc import Callable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from matplotlib_scalebar.scalebar import ScaleBar

from src.defs import Cell
from src.quant.ezmaps.defs import CellMap
from src.quant.ezmaps.rep_map import maps_to_rep_map

RepPlotter = Callable[[Axes, list[Cell], CellMap, dict[str, np.ndarray]], object]


def plot_rep_stratum(
    ax: Axes,
    cells: list[Cell],
    plot_fn: RepPlotter | None = None,
) -> CellMap:
    """Draw an overlay (like foci or intensity), then the representative cell outline."""
    rep_map, transforms = maps_to_rep_map(cells)
    if plot_fn is not None:
        plot_fn(ax, cells, rep_map, transforms)


    rep_map.show(show_mask=False, ax=ax, render=False, scale=1.0, extend_midline_by=0)
    return rep_map


def plot_rep(
    cells: dict[str, Cell],
    plot_fn: RepPlotter | None = None,
    *,
    mappable: ScalarMappable | None = None,
    colorbar_label: str = "",
    n_strata=4,
    outline_margin=0.01,
    subplot_hspace=0.08,
    colorbar_fraction=0.05,
) -> Figure:
    """Plot representative cells grouped by length.
    """
    if n_strata < 1:
        raise ValueError("n_strata must be at least 1")

    cell_list = list(cells.values())
    if not cell_list:
        raise ValueError("at least one cell is required")

    cell_list.sort(
        key=lambda cell: (
            cell.map.midline_length * cell.fov.image_dimensions.microns_per_pixel
        )
    )
    strata = np.array_split(cell_list, min(n_strata, len(cell_list)))
    figure_width = 5.0
    fig_rep, axes = plt.subplots(
        len(strata), 1, figsize=(figure_width, 1), squeeze=False
    )

    rep_strata = []
    for ax, stratum in zip(axes[:, 0], strata):
        rep_strata.append(plot_rep_stratum(ax, list(stratum), plot_fn))
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Use identical limits for every stratum so their outlines are directly
    # comparable.  Add a small margin for the outline.
    all_boundaries = [
        rep_map.boundary_by_arc_length(np.linspace(0, rep_map.perimeter, num=300))
        for rep_map in rep_strata
    ]
    all_geometry = np.concatenate(all_boundaries)
    x_margin = np.ptp(all_geometry[:, 1]) * outline_margin
    y_margin = np.ptp(all_geometry[:, 0]) * outline_margin
    x_limits = (
        np.min(all_geometry[:, 1]) - x_margin,
        np.max(all_geometry[:, 1]) + x_margin,
    )
    y_limits = (
        np.min(all_geometry[:, 0]) - y_margin,
        np.max(all_geometry[:, 0]) + y_margin,
    )
    for ax in axes[:, 0]:
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_xticks([])
        ax.set_yticks([])

    # Size each panel from the data aspect ratio. This makes the figure only
    # as tall as the panels, their gaps, and the small scalebar region require.
    left = 0.04
    right = 0.82
    panel_bottom = 0.55
    panel_width = figure_width * (right - left)
    data_aspect = np.ptp(all_geometry[:, 0]) / np.ptp(all_geometry[:, 1])
    panel_height = panel_width * data_aspect
    gap = subplot_hspace
    figure_height = panel_bottom + len(strata) * panel_height
    figure_height += (len(strata) - 1) * gap + 0.03
    fig_rep.set_size_inches(figure_width, figure_height)

    panel_total_height = len(strata) * panel_height + (len(strata) - 1) * gap
    for index, ax in enumerate(axes[:, 0]):
        y = panel_bottom + (len(strata) - 1 - index) * (panel_height + gap)
        ax.set_position(
            (
                left,
                y / figure_height,
                panel_width / figure_width,
                panel_height / figure_height,
            )
        )

    if mappable is not None:
        colorbar_ax = fig_rep.add_axes(
            (
                0.86,
                panel_bottom / figure_height,
                colorbar_fraction,
                panel_total_height / figure_height,
            )
        )
        colorbar = fig_rep.colorbar(mappable, cax=colorbar_ax)
        colorbar.set_label(colorbar_label, fontsize=11, rotation=270, labelpad=20)
        colorbar.ax.tick_params(labelsize=12)

    scalebar_ax = fig_rep.add_axes(
        (left, 0.04 / figure_height, right - left, 0.18 / figure_height)
    )
    scalebar_ax.set_xlim(x_limits)
    scalebar_ax.set_ylim(0, 1)
    scalebar_ax.axis("off")
    scalebar_ax.add_artist(
        ScaleBar(
            1,
            "um",
            fixed_value=1,
            fixed_units="um",
            color="black",
            box_alpha=0,
            width_fraction=0.06,
            location="lower left",
            rotation="horizontal-only",
            font_properties={"size": 10},
        )
    )

    if fig_rep is None:
        raise Exception("lol, no rep figure was generated, silly")

    return fig_rep
