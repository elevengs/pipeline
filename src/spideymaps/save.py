from pathlib import Path
from src.plot.save import save_foci_plots, save_cell_length_plot
from src.plot.num_foci import plot_foci_counts
from src.spideymaps.spideymaps_v2 import SpideyAtlas
from .make_atlas import atlas_cell_lengths_um, make_atlases
from .plot.save import save_rep_cells_plot
from src.defs import Channel
from src.util.save import save_fig
import matplotlib.pyplot as plt

DEFAULT_POSITION_KEY = "l_abs_centered_pol_um"
DEFAULT_CELL_LENGTH_KEY = "cell_length_um"

def save_spideymaps_channel_plots(
    channel: Channel,
    cell_layers_df,
    atlas: SpideyAtlas,
    out: Path,
):
    """Returns nothing, but saves plots for one finalized SpideyAtlas.
    """
    save_foci_plots(
        cell_layers_df,
        atlas.coords,
        out,
        DEFAULT_POSITION_KEY,
        DEFAULT_CELL_LENGTH_KEY,
    )

    save_cell_length_plot(atlas_cell_lengths_um(atlas), out)

    save_rep_cells_plot(channel,atlas,out)

    fig_foci_counts = plot_foci_counts(cell_layers_df)
    fig_foci_counts.suptitle(channel.name)
    save_fig(fig_foci_counts, out.joinpath("fig_foci_counts"))
    plt.close(fig_foci_counts)

def finalize_channel(
    channel: Channel,
    cell_layers_df,
    atlas: SpideyAtlas,
    out: Path
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    save_spideymaps_channel_plots(channel, cell_layers_df, atlas, out)

    atlas.coords.to_csv(out.joinpath("atlas_coords.csv"), index=False)
    atlas.data.to_csv(
        out.joinpath("atlas_data.csv"),
        index=True,
        index_label=("i_r", "i_l", "i_phi"),
    )

def finalize_channels(
    atlases_by_channel: dict[Channel, SpideyAtlas],
    cell_layers_df,
    out: Path,
) -> None:
    for channel, atlas in atlases_by_channel.items():
        finalize_channel(
            channel,
            cell_layers_df[
                cell_layers_df["LAYER::CHANNEL_NAME"] == channel.name
            ],
            atlas,
            out / channel.name,
        )

def finalize_spideymaps(
    cells,
    cell_layers,
    foci,
    cell_layers_df,
    out: Path,
) -> None:
    _, atlases_by_channel, _ = make_atlases(cells, cell_layers, foci)
    finalize_channels(atlases_by_channel, cell_layers_df, out)
