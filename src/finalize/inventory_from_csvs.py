from collections.abc import Iterable
from pathlib import Path

from tqdm.auto import tqdm
from src.defs import (
    cell_layers_from_csv,
    cells_from_csv,
    foci_from_csv,
    load,
    FOV,
    Cell,
    CellLayer,
    Focus
)
from src.util.misc import csv_path_for
from src.util.parallel import handle_sources_with

def inventory_from_CSVs(
    fov: FOV,

    cells_dir: Path,
    cell_layers_dir: Path,
    foci_dir: Path,
) -> tuple[dict[str, Cell], dict[str, CellLayer], dict[str, Focus]]:
    """Returns the cell, cell layer, and focus records for FOV given pre-computed CSVs.
    """

    cells = cells_from_csv(fov, csv_path_for(fov.source_path, fov.root, cells_dir))
    cell_layers = cell_layers_from_csv(cells, csv_path_for(fov.source_path, fov.root, cell_layers_dir))
    foci = foci_from_csv(cell_layers, csv_path_for(fov.source_path, fov.root, foci_dir))

    return cells, cell_layers, foci

def batch_inventory_from_CSVs(
    sources: Iterable[Path],
    root: Path,
    cells_dir: Path,
    cell_layers_dir: Path,
    foci_dir: Path,
    max_workers: int,
) -> tuple[dict[str, Cell], dict[str, CellLayer], dict[str, Focus]]:
    """Returns dictionaries containing all cells, cell layers, and foci from the sources provided.

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
        lambda source: 
            inventory_from_CSVs(
                load(source, root), 
                cells_dir, 
                cell_layers_dir, 
                foci_dir
            ),
        max_workers,
    )
    for source, result in tqdm(
        results,
        total=len(sources),
        desc="batch inventory from CSVs",
        unit="FOV",
    ):
        if isinstance(result, Exception):
            raise result
        update_data(source, result)
    
    return (cells, cell_layers, foci)