import shapely as sl
from skimage.measure import find_contours
from src.spideymaps.spideymaps_v2 import Spideymap, smooth_skin
from typing import Iterable
from typing import Any

from src.spideymaps.stats import Stats
from src.defs import (
    Channel,
    CellLayer,
    Focus,
    foci_to_dataframe
)

grid_params: dict[str, Any] = dict(
    radius=6,
    n_shells=5,
    n_cols=5,
    n_phi=(1, 3, 5, 5, 5),
    level=0,
)

def map_cell_layer(cell_layer: CellLayer, foci: Iterable[Focus]):
    cell = cell_layer.cell
    foci = list(foci or [])
    stats = Stats()

    num_foci = cell_layer.props.num_foci
    stats.totals_foci[num_foci] = 1
    stats.total_cells = 1

    if num_foci < 1 or num_foci > 3:
        return None, stats
    if len(foci) != num_foci:
        raise ValueError(
            f'cell-layer "{cell_layer.id}" reports '
            f"{num_foci} foci but {len(foci)} were loaded"
        )

    # Now that we've passed the two main conditions,
    # 1) at least one but less than four foci and 
    # 2) availability of a bounding box
    # try to create a map
    stats.total_tried_map_creation = 1

    foci_df = foci_to_dataframe({focus.id: focus for focus in foci})
    cell_mask = cell.map.mask

    try:
        contour = find_contours(cell_mask, level=0.5)
        out = contour[0][:, ::-1]
    except Exception:
        stats.total_failed_contour_creation = 1
        return None, stats

    out = smooth_skin(out, sigma=0.5)
    out = sl.LinearRing(out)

    spidey_map = Spideymap(
        bimage=cell_mask, coords=foci_df,
        xcol="FOCUS::PROPS::DIM_1_COORDINATE", ycol="FOCUS::PROPS::DIM_0_COORDINATE",
    )

    try:
        spidey_map.make_grid(out=out, **grid_params)
        spidey_map.count()
        spidey_map.coords["cell_length"] = spidey_map.mid.length
        spidey_map.microns_per_pixel = cell.fov.image_dimensions.microns_per_pixel  # ty: ignore[unresolved-attribute]
        stats.total_passed_map_creation = 1
    except Exception as _:
        stats.total_failed_grid_creation = 1
        return None, stats

    return spidey_map, stats

def batch_map(
    cell_layers: dict[str, CellLayer],
    foci: dict[str, Focus]
) -> tuple[dict[Channel, list[Spideymap]], dict[Channel, Stats]]:
    """Returns ``(combined, by_channel)`` where
    ``combined`` contains the atlas coords for every focus,
    and ``by_channel`` contains the atlas coords for each focus in a given channel.
    """
    foci_by_cell_layer_id: dict[str, list[Focus]] = {}
    for _, focus in foci.items():
        foci_by_cell_layer_id.setdefault(
            focus.cell_layer.id, []
        ).append(focus)

    maps_by_channel: dict[Channel, list[Spideymap]] = {}
    stats_by_channel: dict[Channel, Stats] = {}
    
    for _, cell_layer in cell_layers.items():
        map, stats = map_cell_layer(
            cell_layer,
            foci_by_cell_layer_id.get(cell_layer.id, [])
        )
        channel = cell_layer.layer.channel

        if map is not None:
            maps_by_channel.setdefault(channel, []).append(map)
        stats_by_channel[channel] = stats_by_channel.get(channel, Stats()) + stats

    return maps_by_channel, stats_by_channel
