
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colorizer import Colorizer
from matplotlib.figure import Figure
from matplotlib_scalebar.scalebar import ScaleBar
from pandas import DataFrame
from scipy.stats import gaussian_kde

from src.defs import Cell
from src.quant.ezmaps.rep_map import maps_to_rep_map


def plot_rep_stratum(
    ax,
    cells: list[Cell],
    foci_df: DataFrame,
    coordinate_0_key: str,
    coordinate_1_key: str,
    cmap,
    colorizer,
):
    """Plot one representative cell and its foci on ``ax``."""
    rep_map, transforms = maps_to_rep_map(cells)

    rep_points = []
    for _, focus in foci_df.iterrows():
        cell_id = str(focus["CELL::ID"])
        if cell_id not in transforms:
            continue

        vector_from_centroid = np.asarray(
            [focus[coordinate_0_key], focus[coordinate_1_key]],
            dtype=float,
        )
        if not np.all(np.isfinite(vector_from_centroid)):
            continue

        rep_points.append(transforms[cell_id] @ vector_from_centroid)

    if rep_points:
        points = np.asarray(rep_points)[:, [1, 0]].T
        try:
            density = gaussian_kde(points)(points)
        except (np.linalg.LinAlgError, ValueError):
            density = np.ones(points.shape[1])

        order = np.argsort(density)
        ax.scatter(
            points[0, order],
            points[1, order],
            c=density[order],
            s=5,
            edgecolor="none",
            colorizer=colorizer,
        )

    # Draw the outline after the points for visibility.
    rep_map.show(show_mask=False, ax=ax, render=False, scale=1.0, extend_midline_by=0)
    return rep_map, np.asarray(rep_points)


def plot_rep(
    cells: dict[str, Cell],
    cell_layers_df: DataFrame,
    foci_df: DataFrame,
    out: Path,
    coordinate_0_key: str = "FOCUS::VECTOR_FROM_CENTROID_COORD_0_UM",
    coordinate_1_key: str = "FOCUS::VECTOR_FROM_CENTROID_COORD_1_UM",
    cmap=sns.color_palette("plasma", as_cmap=True),
    n_strata=4,
    outline_margin=0.01,
    subplot_hspace=0.08,
    colorbar_fraction=0.05,
) -> Figure:
    if n_strata < 1:
        raise ValueError("n_strata must be at least 1")

    cell_list = list(cells.values())
    if not cell_list:
        raise ValueError("at least one cell is required")

    cell_list.sort(
        key=lambda cell: cell.map.midline_length
        * cell.fov.image_dimensions.microns_per_pixel
    )
    strata = np.array_split(cell_list, min(n_strata, len(cell_list)))
    norm = plt.Normalize(vmin=0, vmax=4)
    colorizer = Colorizer(cmap=cmap, norm=norm)
    figure_width = 5.0
    fig_rep, axes = plt.subplots(len(strata), 1, figsize=(figure_width, 1), squeeze=False)

    rep_strata = []
    for ax, stratum in zip(axes[:, 0], strata):
        rep_strata.append(
            plot_rep_stratum(
            ax,
            list(stratum),
            foci_df,
            coordinate_0_key,
            coordinate_1_key,
            cmap,
            colorizer,
            )
        )
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Use identical limits for every stratum so their outlines are directly
    # comparable.  Add a small margin for the outline and focus points.
    all_boundaries = [
        rep_map.boundary_by_arc_length(
            np.linspace(0, rep_map.perimeter, num=300)
        )
        for rep_map, _ in rep_strata
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
    panel_bottom = 0.10
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

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar_ax = fig_rep.add_axes(
        (0.86, panel_bottom / figure_height, colorbar_fraction, panel_total_height / figure_height)
    )
    colorbar = fig_rep.colorbar(scalar_mappable, cax=colorbar_ax)
    colorbar.set_label("Probability density", fontsize=11, rotation=270, labelpad=20)
    colorbar.ax.tick_params(labelsize=12)

    scalebar_ax = fig_rep.add_axes(
        (left, 0.005 / figure_height, 0.25, 0.06 / figure_height)
    )
    scalebar_ax.set_xlim(0, 2)
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
            location="center",
            rotation="horizontal-only",
            font_properties={"size": 10},
        )
    )

    if fig_rep is None:
        raise Exception("lol, no rep figure was generated, silly")

    return fig_rep
