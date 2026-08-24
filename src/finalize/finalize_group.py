from dataclasses import dataclass
from pathlib import Path

from src.defs import load
from src.plot.save import (
    save_all_plots,
    save_all_spideymaps_plots,
    save_cell_length_plot,
)
from src.spideymaps.make_atlas import (
    channel_output_dir,
    channel_output_dir_for_name,
    make_atlas_coords,
)

from .make_final_dfs import make_final_dfs

@dataclass(frozen=True)
class SpideymapsOptions:
    """Inputs required by the existing Spideymaps atlas API.
    """

    root: Path
    cells_dir: Path
    cell_layers_dir: Path
    foci_dir: Path
    out: Path
    max_workers: int


def group_output_dir(out: Path, group: str) -> Path:
    """Returns the output directory for one FOV group.
    """
    return out / group


def sources_by_group(sources: list[Path], root: Path) -> dict[str, list[Path]]:
    """Returns a ``dict`` with metadata-defined experimental groups as keys and lists of sources as values.
    """
    result: dict[str, list[Path]] = {}
    for source in sources:
        fov = load(source, root)
        result.setdefault(fov.group, []).append(source)
    return result


def make_group_outputs(
    sources: list[Path],
    root: Path,
    cells_dir: Path,
    cell_layers_dir: Path,
    foci_dir: Path,
    out: Path,
    min_bio_reps: int,
    max_workers: int,
    enable_spideymaps: bool,
) -> None:
    """Returns nothing, but creates finalized outputs for one metadata-defined group.
    """
    cells_df, cell_layers_df, foci_by_channel = make_final_dfs(
        sources,
        root,
        cells_dir,
        cell_layers_dir,
        foci_dir,
        out,
        min_bio_reps,
        max_workers,
    )
    save_cell_length_plot(cells_df, out)

    for channel_name, foci_df in foci_by_channel.items():
        channel_out = channel_output_dir_for_name(out, channel_name)
        channel_cell_layers_df = cell_layers_df[
            cell_layers_df["LAYER::CHANNEL_NAME"] == channel_name
        ]
        save_all_plots(cells_df, channel_cell_layers_df, foci_df, channel_out)

    if not enable_spideymaps:
        return

    spideymaps_out = out / "spideymaps"
    options = SpideymapsOptions(
        root=root,
        cells_dir=cells_dir,
        cell_layers_dir=cell_layers_dir,
        foci_dir=foci_dir,
        out=spideymaps_out,
        max_workers=max_workers,
    )
    coords_by_channel = make_atlas_coords(sources, options)
    for channel, coords in coords_by_channel.items():
        channel_out = channel_output_dir(spideymaps_out, channel)
        channel_cell_layers_df = cell_layers_df[
            cell_layers_df["LAYER::CHANNEL_NAME"] == channel.name
        ]
        save_all_spideymaps_plots(
            cells_df,
            channel_cell_layers_df,
            coords,
            channel_out,
        )
