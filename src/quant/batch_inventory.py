from pathlib import Path
import traceback

from tqdm.auto import tqdm

from src.defs import load
from src.quant.inventory import write_inventory_CSVs
from .focus_detection.config import FocusDetectionConfig
from src.util.incl_excl import select_files
from src.util.misc import replace_root_with, csv_path_for
from src.util.parallel import handle_sources_with


def batch_inventory(
    root: Path,
    include: list[str],
    exclude: list[str],
    focus_detection_config: FocusDetectionConfig,
    cells_dir: Path,
    cell_layers_dir: Path,
    foci_dir: Path,
    stepwise_figures_dir: Path | None = None,
    max_workers=5,
):

    DESC = "Batch inventory"

    sources = list(
        select_files(
            root,
            include=include,
            exclude=exclude,
        )
    )

    def source_handler(source):
        fov = load(source, root)
        cells_dest = csv_path_for(fov.source_path, root, cells_dir)
        cell_layers_dest = csv_path_for(fov.source_path, root, cell_layers_dir)
        foci_dest = csv_path_for(fov.source_path, root, foci_dir)
        stepwise_figures_path = (
            None
            if stepwise_figures_dir is None
            else replace_root_with(fov.source_path, root, stepwise_figures_dir)
        )
        write_inventory_CSVs(
            fov,
            cells_dest,
            cell_layers_dest,
            foci_dest,
            focus_detection_config,
            stepwise_figures_path,
        )

    results = handle_sources_with(sources, source_handler, max_workers)

    with tqdm(total=len(sources), desc=DESC, unit="FOV") as pbar:
        for source, result in results:
            try:
                if isinstance(result, Exception):
                    raise result
            except Exception:
                tqdm.write(f"FAILED: {source}")
                tqdm.write(traceback.format_exc())
            finally:
                pbar.update(1)
