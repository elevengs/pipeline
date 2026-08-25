from src.spideymaps.spideymaps_v2 import SpideyAtlas
from .rep_cells import plot_rep_cells
from src.defs import Channel
from src.util.save import save_fig
import matplotlib.pyplot as plt
from pathlib import Path

def save_rep_cells_plot(
    channel: Channel, 
    atlas: SpideyAtlas,
    out: Path
):
    fig_maps, _ = plot_rep_cells(channel, atlas)
    fig_maps.suptitle(channel.name)
    save_fig(fig_maps, out.joinpath("fig_maps"))
    plt.close(fig_maps)
