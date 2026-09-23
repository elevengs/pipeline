from pathlib import Path

from src.defs import load
from src.plot.save import (
    save_all_plots,
    save_cell_length_plot,
)
from src.spideymaps.save import finalize_spideymaps
from .inventory_from_csvs import batch_inventory_from_CSVs
from .make_final_dfs import make_final_dfs


def group_output_dir(out: Path, group: str) -> Path:
    """Returns the output directory for one FOV group."""
    return out / group


def sources_by_group(sources: list[Path], root: Path) -> dict[str, list[Path]]:
    """Returns a ``dict`` with metadata-defined experimental groups as keys and lists of sources as values."""
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
    """Returns nothing, but creates finalized outputs for one metadata-defined group."""

    cells, cell_layers, foci = batch_inventory_from_CSVs(
        sources, root, cells_dir, cell_layers_dir, foci_dir, max_workers
    )

    cells_df, cell_layers_df, foci_by_channel = make_final_dfs(
        cells,
        cell_layers,
        foci,
        out,
        min_bio_reps,
        max_workers,
    )

    save_cell_length_plot(cells_df["CELL::PROPS::AXIS_MAJOR_LENGTH"], out)

    for channel_name, foci_df in foci_by_channel.items():
        channel_out = out / channel_name
        channel_cell_layers_df = cell_layers_df[
            cell_layers_df["LAYER::CHANNEL_NAME"] == channel_name
        ]
        channel_foci_df = foci_df[
            foci_df["LAYER::CHANNEL_NAME"] == channel_name
        ]
        save_all_plots(
            cells,
            cells_df,
            channel_cell_layers_df,
            channel_foci_df,
            channel_out
        )

    if not enable_spideymaps:
        return

    spideymaps_out = out / "spideymaps"

    finalize_spideymaps(
        cells,
        cell_layers,
        foci,
        cell_layers_df,
        spideymaps_out,
    )
