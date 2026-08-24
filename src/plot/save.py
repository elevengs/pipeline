from matplotlib import pyplot as plt
from pathlib import Path

from src.util.misc import mkdir_p
from src.util.save import save_fig
from .cell_length_hist import DEFAULT_CELL_LENGTH_KEY, plot_cell_length
from .num_foci import plot_foci_counts
from .volcano.main import DEFAULT_POSITION_KEY, plot_volcano
from .volcano.intensity import scatter_intensity
from .volcano.kde import scatter_kde_colors
from .volcano.num_foci import scatter_num_foci


def save_foci_plots(
    cell_layers_df,
    foci_df,
    out: Path,
    position_key,
    cell_length_key,
):
    fig_cano = plot_volcano(
        [foci_df],
        scatter_fn=scatter_kde_colors,
        position_key=position_key,
        cell_length_key=cell_length_key,
    )
    fig_foci = plot_volcano(
        [foci_df],
        scatter_fn=scatter_num_foci,
        position_key=position_key,
        cell_length_key=cell_length_key,
    )
    fig_intensity = plot_volcano(
        [foci_df],
        scatter_fn=scatter_intensity,
        position_key=position_key,
        cell_length_key=cell_length_key,
    )
    fig_num_foci = plot_foci_counts(cell_layers_df)

    save_folder = out
    mkdir_p(save_folder)
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


def save_cell_length_plot(dataframe, out, cell_length_key=DEFAULT_CELL_LENGTH_KEY):
    if "CELL::ID" in dataframe.columns:
        dataframe = dataframe.drop_duplicates("CELL::ID")

    fig = plot_cell_length(dataframe, cell_length_key)
    mkdir_p(out)
    save_fig(fig, out.joinpath("cell_length"))
    plt.close(fig)


def save_all_plots(
    cells_df,
    cell_layers_df,
    foci_df,
    out: Path,
    position_key=DEFAULT_POSITION_KEY,
    cell_length_key="CELL::MIDLINE_LENGTH_UM",
):
    save_foci_plots(
        cell_layers_df,
        foci_df,
        out,
        position_key,
        cell_length_key,
    )


def save_all_spideymaps_plots(
    cells_df,
    cell_layers_df,
    atlas_coords_df,
    out: Path,
    position_key="l_abs_centered_pol_um",
    cell_length_key="cell_length_um",
):
    """Returns nothing, but saves plots for one finalized Spideymaps atlas dataframe.
    """
    save_foci_plots(
        cell_layers_df,
        atlas_coords_df,
        out,
        position_key,
        cell_length_key,
    )

    spideymaps_cell_lengths = atlas_coords_df[["CELL::ID", cell_length_key]].rename(
        columns={cell_length_key: DEFAULT_CELL_LENGTH_KEY}
    )
    save_cell_length_plot(
        spideymaps_cell_lengths,
        out,
    )
