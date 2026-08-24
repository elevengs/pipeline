from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from src.defs import (
    cell_layers_from_csv,
    cell_layers_to_dataframe,
    cells_from_csv,
    cells_to_dataframe,
    foci_from_csv,
    foci_to_dataframe,
    load,
    Cell,
    CellLayer,
    Focus
)
from src.util.misc import csv_path_for
from src.util.parallel import handle_sources_with


def source_handler(
    source: Path,
    root: Path,
    cells_dir: Path,
    cell_layers_dir: Path,
    foci_dir: Path,
) -> tuple[dict[str, Cell], dict[str, CellLayer], dict[str, Focus]]:
    """Returns the cell, cell layer, and focus records for one source image.
    """
    fov = load(source, root)

    cells = cells_from_csv(fov, csv_path_for(source, root, cells_dir))
    cell_layers = cell_layers_from_csv(
        cells,
        csv_path_for(source, root, cell_layers_dir),
    )
    foci = foci_from_csv(cell_layers, csv_path_for(source, root, foci_dir))

    return cells, cell_layers, foci


def validate_biological_replicates(
    name: str,
    dataframe: pd.DataFrame,
    minimum: int,
) -> None:
    """Require images from at least ``minimum`` distinct dates."""
    unique_dates = pd.unique(dataframe["FOV::DATE"])

    if len(unique_dates) < minimum:
        raise ValueError(
            f"expected {name} from at least {minimum} biological replicates "
            f"(i.e. images from different dates) but found just "
            f"{len(unique_dates)}: {unique_dates}"
        )


def make_final_dfs(
    sources: Iterable[Path],
    root: Path,
    cells_dir: Path,
    cell_layers_dir: Path,
    foci_dir: Path,
    out: Path,
    min_bio_reps: int,
    max_workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Returns unified cells and cell layers ``DataFrame``s, as well as a dictionary of foci ``DataFrame``s keyed by channel names.

    Combines per-source quantification CSVs and write finalized CSVs.
    """
    cells = {}
    cell_layers = {}
    foci = {}
    sources = list(sources)

    def update_data(source: Path, result) -> None:
        source_cells, source_cell_layers, source_foci = result
        cells.update(source_cells)
        cell_layers.update(source_cell_layers)
        foci.update(source_foci)

    results = handle_sources_with(
        sources,
        lambda source: source_handler(
            source, root, cells_dir, cell_layers_dir, foci_dir
        ),
        max_workers,
    )
    for source, result in tqdm(
        results,
        total=len(sources),
        desc="Reading cells and foci from CSVs",
        unit="FOV",
    ):
        if isinstance(result, Exception):
            raise result
        update_data(source, result)

    cells_df = cells_to_dataframe(cells)
    cell_layers_df = cell_layers_to_dataframe(cell_layers)
    foci_df = foci_to_dataframe(foci)

    validate_biological_replicates("cells", cells_df, min_bio_reps)
    validate_biological_replicates("cell layers", cell_layers_df, min_bio_reps)
    validate_biological_replicates("foci", foci_df, min_bio_reps)

    out.mkdir(parents=True, exist_ok=True)
    cells_df.to_csv(out / "cells.csv", index=False)
    cell_layers_df.to_csv(out / "cell_layers.csv", index=False)
    foci_df.to_csv(out / "foci.csv", index=False)

    foci_by_channel = {
        channel_name: channel_foci_df.reset_index(drop=True)
        for channel_name, channel_foci_df in foci_df.groupby("LAYER::CHANNEL_NAME")
    }
    return cells_df, cell_layers_df, foci_by_channel  # ty: ignore[invalid-return-type]
