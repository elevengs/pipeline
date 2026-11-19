from functools import partial
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from .config import DEFAULT_PLOT_CONFIG, PlotConfig

from pandas import DataFrame

from src.defs import Cell
from src.plot.rep.foci import plot_rep_foci
from src.plot.rep.intensity import plot_rep_intensity
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
    plot_config: PlotConfig = DEFAULT_PLOT_CONFIG,
):
    fig_rep_foci = plot_rep_foci(
        cells,
        foci_df,
        symmetrize=plot_config.rep_foci_symmetrize,
        n_strata=plot_config.rep_foci_n_strata,
        scalebar_size_um=plot_config.rep_foci_scalebar_size_um,
        average_method=plot_config.rep_foci_average_method,
        colorbar_min=plot_config.rep_foci_colorbar_min,
        colorbar_max=plot_config.rep_foci_colorbar_max,
    )
    fig_cano = plot_lengthwise_scatter(
        [foci_df],
        scatter_fn=partial(
            scatter_kde_colors,
            colorbar_min=plot_config.lengthwise_scatter_kde_colorbar_min,
            colorbar_max=plot_config.lengthwise_scatter_kde_colorbar_max,
        ),
        min_plot_len=plot_config.lengthwise_scatter_kde_y_min,
        max_plot_len=plot_config.lengthwise_scatter_kde_y_max,
        position_key=position_key,
        cell_length_key=cell_length_key,
    )
    fig_foci = plot_lengthwise_scatter(
        [foci_df],
        scatter_fn=partial(
            scatter_num_foci,
            colorbar_min=plot_config.lengthwise_scatter_num_foci_colorbar_min,
            colorbar_max=plot_config.lengthwise_scatter_num_foci_colorbar_max,
        ),
        min_plot_len=plot_config.lengthwise_scatter_num_foci_y_min,
        max_plot_len=plot_config.lengthwise_scatter_num_foci_y_max,
        position_key=position_key,
        cell_length_key=cell_length_key,
    )
    fig_intensity = plot_lengthwise_scatter(
        [foci_df],
        scatter_fn=partial(
            scatter_intensity,
            colorbar_min=plot_config.lengthwise_scatter_intensity_colorbar_min,
            colorbar_max=plot_config.lengthwise_scatter_intensity_colorbar_max,
        ),
        min_plot_len=plot_config.lengthwise_scatter_intensity_y_min,
        max_plot_len=plot_config.lengthwise_scatter_intensity_y_max,
        position_key=position_key,
        cell_length_key=cell_length_key,
    )
    fig_num_foci = plot_foci_counts(cell_layers_df)

    save_folder = out
    mkdir_p(save_folder)
    save_fig(fig_rep_foci, save_folder.joinpath("rep_foci"))
    save_fig(fig_cano, save_folder.joinpath("long_axis_position_vs_cell_length"))
    save_fig(fig_foci, save_folder.joinpath("long_axis_position_vs_foci_number"))
    save_fig(
        fig_intensity, save_folder.joinpath("long_axis_position_vs_foci_intensity")
    )
    save_fig(fig_num_foci, save_folder.joinpath("num_foci"))

    plt.close(fig_rep_foci)
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
    channel_name: str,
    cells,
    cells_df,
    cell_layers_df,
    foci_df,
    out: Path,
    position_key=DEFAULT_POSITION_KEY,
    cell_length_key="CELL::MIDLINE_LENGTH_UM",
    plot_config: PlotConfig = DEFAULT_PLOT_CONFIG,
):
    fig_rep_intensity = plot_rep_intensity(
        cells,
        channel_name,
        symmetrize=plot_config.rep_intensity_symmetrize,
        n_strata=plot_config.rep_intensity_n_strata,
        scalebar_size_um=plot_config.rep_intensity_scalebar_size_um,
        average_method=plot_config.rep_intensity_average_method,
        norm=Normalize(
            vmin=plot_config.rep_intensity_colorbar_min,
            vmax=plot_config.rep_intensity_colorbar_max,
        ),
    )
    mkdir_p(out)
    save_fig(fig_rep_intensity, out.joinpath("rep_intensity"))
    plt.close(fig_rep_intensity)
    save_foci_plots(
        cells,
        cell_layers_df,
        foci_df,
        out,
        position_key,
        cell_length_key,
        plot_config=plot_config,
    )
