from typing import Iterable
import pandas as pd
from src.spideymaps.spideymaps_v2 import Spideymap

from .map import batch_map

from src.spideymaps.spideymaps_v2 import SpideyAtlas
from src.spideymaps.stats import Stats
from src.defs import (
    Channel,
    Cell,
    CellLayer,
    Focus,
)

def add_polarity(atlas_coords):
    atlas_coords["l_abs_centered"] = (
        atlas_coords["l_abs"] - atlas_coords["cell_length"] / 2
    )

    # Set the cell's pole to align with the mean location of its foci
    atlas_coords["polarity"] = atlas_coords.groupby("CELL::ID")[
        "l_abs_centered"
    ].transform(lambda x: 1 if x.mean() > 0 else -1)

    atlas_coords["l_abs_centered_pol"] = (
        atlas_coords["l_abs_centered"] * atlas_coords["polarity"]
    )

    atlas_coords["l_abs_centered_pol_um"] = (
        atlas_coords["l_abs_centered_pol"] * atlas_coords["FOV::MICRONS_PER_PIXEL"]
    )

def make_channel_atlas(
    channel: Channel,
    maps: list[Spideymap],
    stats: Stats,
) -> SpideyAtlas | None:

    if not maps:
        return None

    atlas = SpideyAtlas(maps)

    atlas.get_colicoords()

    atlas.coords["cell_length_um"] = (atlas.coords["cell_length"] * atlas.coords["FOV::MICRONS_PER_PIXEL"])

    add_polarity(atlas.coords)

    return atlas

def atlas_cell_lengths_um(atlas: SpideyAtlas) -> Iterable[float]:
    return [
        spideymap.mid.length * spideymap.microns_per_pixel
        for spideymap in atlas.maps.values()
    ]

def make_atlases(
    cells: dict[str, Cell],
    cell_layers: dict[str, CellLayer],
    foci: dict[str, Focus]
) -> tuple[pd.DataFrame, dict[Channel, SpideyAtlas], dict[Channel, Stats]]:
    """Returns ``(combined, by_channel)`` where
    ``combined`` contains the atlas coords for every focus,
    and ``by_channel`` contains the atlas for each focus in a given channel.
    """

    maps_by_channel, stats_by_channel = batch_map(
        cell_layers,
        foci
    )
    
    channels = sorted(
        stats_by_channel,
        key=lambda channel: channel.name,
    )

    atlases_by_channel = {
        channel: make_channel_atlas(
            channel,
            maps_by_channel.get(channel, []),
            stats_by_channel[channel]
        )
        for channel in channels
    }
    atlases_by_channel = {
        channel: atlas for channel, atlas in atlases_by_channel.items()
        if atlas is not None
    }

    combined_coords: pd.DataFrame = (
        pd.concat(
            [atlas.coords for atlas in atlases_by_channel.values()],
            ignore_index=True,
        )
        if atlases_by_channel
        else pd.DataFrame()
    )

    return combined_coords, atlases_by_channel, stats_by_channel
