from types import SimpleNamespace
from pathlib import Path

import pandas as pd

from ..cell import Cell
from ..records import objects_to_dataframe
from .main import CellLayer, index_by_id


def cell_layers_to_dataframe(
    cell_layers: dict[str, CellLayer],
) -> pd.DataFrame:
    return objects_to_dataframe(
        cell_layers.values(),
        CellLayer.csv_column_names(),
        CellLayer.csv_column_values,
        "CELL_LAYER::PROPS::",
    )


def cell_layers_from_dataframe(
    cells: dict[str, Cell],
    cell_layers_df: pd.DataFrame,
) -> dict[str, CellLayer]:
    cells_by_id = index_by_id(cells)
    result = {}

    for _, row in cell_layers_df.iterrows():
        cell_id = str(row["CELL::ID"])
        if cell_id not in cells_by_id:
            raise ValueError(f'cell ID "{cell_id}" was not found')

        cell = cells_by_id[cell_id]

        layers_by_index = {layer.index: layer for layer in cell.fov.layers}
        layer_index = int(row["LAYER::INDEX"])
        if layer_index not in layers_by_index:
            raise ValueError(f"layer index {layer_index} was not found in FOV")
        layer = layers_by_index[layer_index]

        channel = row["LAYER::CHANNEL_NAME"]
        if str(row["LAYER::CHANNEL_NAME"]) != layer.channel.name:
            raise ValueError(
                f'layer index {layer_index} contains channel "{layer.channel.name}", '
                f'not "{channel}"'
            )

        cell_layer = CellLayer(
            cell,
            layer,
            SimpleNamespace(
                **{
                    column.removeprefix("CELL_LAYER::PROPS::").lower(): value
                    for column, value in row.items()
                    if column.startswith("CELL_LAYER::PROPS::") and not pd.isna(value)
                }
            ),
        )
        row_id = str(row["CELL_LAYER::ID"])
        if row_id != cell_layer.id:
            raise ValueError(
                f'cell-layer ID "{row_id}" does not match reconstructed ID '
                f'"{cell_layer.id}"'
            )
        if cell_layer.id in result:
            raise ValueError(f'cell-layer ID "{cell_layer.id}" appears more than once')
        result[cell_layer.id] = cell_layer

    return result


def cell_layers_from_csv(
    cells: dict[str, Cell],
    csv_path: Path,
) -> dict[str, CellLayer]:
    cell_layers_df = pd.read_csv(csv_path)
    return cell_layers_from_dataframe(cells, cell_layers_df)
