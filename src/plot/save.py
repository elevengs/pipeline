from matplotlib import pyplot as plt
from pathlib import Path

from pandas import DataFrame

from src.defs import Cell
from src.plot.rep.main import plot_rep
from src.util.misc import mkdir_p
from src.util.save import save_fig
from .cell_length_hist import plot_cell_length
from .num_foci import plot_foci_counts
from .lengthwise_scatter.main import DEFAULT_POSITION_KEY, plot_lengthwise_scatter
from .lengthwise_scatter.intensity import scatter_intensity
from .lengthwise_scatter.kde import scatter_kde_colors
from .lengthwise_scatter.num_foci import scatter_num_foci


def save_foci_plots(
    cells: dict[str, Cell],
    cell_layers_df: DataFrame,
    foci_df: DataFrame,
    out: Path,
    position_key: str,
    cell_length_key: str,
):
    fig_rep = plot_rep(
        cells,
        cell_layers_df,
        foci_df,
        out,
    )
    fig_cano = plot_lengthwise_scatter(
        [foci_df],
        scatter_fn=scatter_kde_colors,
        position_key=position_key,
        cell_length_key=cell_length_key,
    )
    fig_foci = plot_lengthwise_scatter(
        [foci_df],
        scatter_fn=scatter_num_foci,
        position_key=position_key,
        cell_length_key=cell_length_key,
    )
    fig_intensity = plot_lengthwise_scatter(
        [foci_df],
        scatter_fn=scatter_intensity,
        position_key=position_key,
        cell_length_key=cell_length_key,
    )
    fig_num_foci = plot_foci_counts(cell_layers_df)

    save_folder = out
    mkdir_p(save_folder)
    save_fig(fig_rep, save_folder.joinpath("rep"))
    save_fig(fig_cano, save_folder.joinpath("long_axis_position_vs_cell_length"))
    save_fig(fig_foci, save_folder.joinpath("long_axis_position_vs_foci_number"))
    save_fig(
        fig_intensity, save_folder.joinpath("long_axis_position_vs_foci_intensity")
    )
    save_fig(fig_num_foci, save_folder.joinpath("num_foci"))

    plt.close(fig_cano)
    plt.close(fig_foci)
    plt.close(fig_intensity)
    plt.close(fig_num_foci)


def save_cell_length_plot(lengths, out):
    fig = plot_cell_length(lengths)
    mkdir_p(out)
    save_fig(fig, out.joinpath("cell_length"))
    plt.close(fig)


def save_all_plots(
    cells,
    cells_df,
    cell_layers_df,
    foci_df,
    out: Path,
    position_key=DEFAULT_POSITION_KEY,
    cell_length_key="CELL::MIDLINE_LENGTH_UM",
):
    save_foci_plots(
        cells,
        cell_layers_df,
        foci_df,
        out,
        position_key,
        cell_length_key,
    )
