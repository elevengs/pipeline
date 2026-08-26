from pathlib import Path

import pandas as pd

from src.quant.ezmaps import CellPoint

from ..cell_layer import CellLayer
from ..records import objects_to_dataframe, row_to_properties
from .main import Focus


def foci_to_dataframe(foci: dict[str, Focus]) -> pd.DataFrame:
    return objects_to_dataframe(
        foci.values(),
        Focus.csv_column_names(),
        Focus.csv_column_values,
        "FOCUS::PROPS::",
    )


def foci_from_dataframe(
    cell_layers: dict[str, CellLayer],
    foci_df: pd.DataFrame,
) -> dict[str, Focus]:

    foci = {}
    for _, row in foci_df.iterrows():
        cell_layer_id = str(row["CELL_LAYER::ID"])

        if cell_layer_id not in cell_layers:
            raise ValueError(f'cell layer ID "{cell_layer_id}" was not found')

        cell_layer = cell_layers[cell_layer_id]
        if str(row["CELL::ID"]) != cell_layer.cell.id:
            raise ValueError(
                f'cell layer "{cell_layer_id}" belongs to cell '
                f'"{cell_layer.cell.id}", not "{row["CELL::ID"]}"'
            )

        layer_index = int(row["LAYER::INDEX"])
        if layer_index != cell_layer.layer.index:
            raise ValueError(
                f'cell layer "{cell_layer_id}" uses layer index '
                f"{cell_layer.layer.index}, not {layer_index}"
            )

        if str(row["LAYER::CHANNEL_NAME"]) != cell_layer.layer.channel.name:
            raise ValueError(
                f"layer index {layer_index} contains channel "
                f'"{cell_layer.layer.channel.name}", not "{row["LAYER::CHANNEL_NAME"]}"'
            )

        cell_point = CellPoint(
            midline_position_px=row["FOCUS::MIDLINE_POSITION_PX"],  # type: ignore
            offset_from_midline_px=row["FOCUS::OFFSET_FROM_MIDLINE_PX"],  # type: ignore
        )

        focus = Focus(cell_layer, int(row["FOCUS::INDEX"]), cell_point)  # type: ignore

        focus.props = row_to_properties(row, "FOCUS::PROPS::")

        row_focus_id = row["FOCUS::ID"]

        if str(row_focus_id) != focus.id:
            raise ValueError(
                f'focus ID "{row_focus_id}" does not match reconstructed ID '
                f'"{focus.id}"'
            )

        if focus.id in foci:
            raise ValueError(f'focus ID "{focus.id}" appears more than once')
        foci[focus.id] = focus

    return foci


def foci_from_csv(
    cell_layers: dict[str, CellLayer],
    csv_path: Path,
) -> dict[str, Focus]:
    foci_df = pd.read_csv(csv_path)
    return foci_from_dataframe(cell_layers, foci_df)
