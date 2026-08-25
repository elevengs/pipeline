from src.defs import Cell, CellLayer, Focus
from pathlib import Path

import pandas as pd

from src.defs import (
    cell_layers_to_dataframe,
    cells_to_dataframe,
    foci_to_dataframe,
)

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
    cells: dict[str, Cell],
    cell_layers: dict[str, CellLayer],
    foci: dict[str, Focus],
    out: Path,
    min_bio_reps: int,
    max_workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Returns unified cells and cell layers ``DataFrame``s, as well as a dictionary of foci ``DataFrame``s keyed by channel names.

    Combines per-source quantification CSVs and write finalized CSVs.
    """
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
