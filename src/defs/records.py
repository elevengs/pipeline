from types import SimpleNamespace
from collections.abc import Callable, Iterable

import pandas as pd


def objects_to_dataframe(
    objects: Iterable,
    column_names: list[str],
    column_values: Callable,
    properties_prefix: str,
) -> pd.DataFrame:
    """Returns DataFrame representing objects with a ``props`` namespace, such as ``Cell``, `CellLayer``, and ``Focus``.
    """
    objects = list(objects)
    columns = list(column_names)
    property_columns = sorted(
        {
            f"{properties_prefix}{name.upper()}"
            for obj in objects
            for name in vars(obj.props)
        }
    )
    columns.extend(column for column in property_columns if column not in columns)

    rows = []
    for obj in objects:
        values = dict(zip(column_names, column_values(obj)))
        values.update(
            {
                f"{properties_prefix}{name.upper()}": value
                for name, value in vars(obj.props).items()
            }
        )
        rows.append(values)

    return pd.DataFrame(rows, columns=columns)

def row_to_properties(
    row: pd.Series,
    properties_prefix: str
) -> SimpleNamespace:
    res = SimpleNamespace()
    for column, value in row.items():
        if not isinstance(column, str) or not column.startswith(properties_prefix) or pd.isna(value):
            continue
        setattr(res, column.removeprefix(properties_prefix).lower(), value)
    return res
    