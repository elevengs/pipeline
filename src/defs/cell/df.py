import pandas as pd
from pathlib import Path

from ..fov import FOV
from ..records import objects_to_dataframe
from .main import Cell


def cells_to_dataframe(cells: dict[str, Cell]) -> pd.DataFrame:
    return objects_to_dataframe(
        cells.values(),
        Cell.csv_column_names(),
        Cell.csv_column_values,
        "CELL::PROPS::",
    )


def cells_from_dataframe(fov: FOV, cells_df: pd.DataFrame) -> dict[str, Cell]:
    cells = {}
    for _, row in cells_df.iterrows():
        label_column = "CELL::LABEL"
        id_column = "CELL::ID"
        cell = Cell(
            fov,
            int(row[label_column]),
            str(row[id_column]),
        )
        for column, value in row.items():
            prefix = "CELL::PROPS::"
            if not column.startswith(prefix) or pd.isna(value):
                continue
            setattr(cell.props, column.removeprefix(prefix).lower(), value)
        cells[cell.id] = cell

    return cells


def cells_from_csv(fov: FOV, csv_path: Path) -> dict[str, Cell]:
    cells_df = pd.read_csv(csv_path)
    return cells_from_dataframe(fov, cells_df)
