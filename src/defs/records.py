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
