from types import SimpleNamespace

from ..cell import Cell
from ..fov import Layer


def index_by_id(items):
    if isinstance(items, dict):
        items = items.values()
    items_by_id = {}
    for item in items:
        if item.id in items_by_id:
            raise ValueError(f'duplicate ID "{item.id}"')
        items_by_id[item.id] = item
    return items_by_id


class CellLayer:
    id: str
    cell: Cell
    layer: Layer
    props: SimpleNamespace

    @property
    def cell_id(self):
        return self.cell.id

    @property
    def label(self):
        return self.cell.label

    @property
    def fov(self):
        return self.cell.fov

    def __init__(self, cell: Cell, layer: Layer, props: SimpleNamespace | None = None):
        self.id = f"{cell.id}_{layer.channel.name}"
        self.cell = cell
        self.layer = layer
        self.props = SimpleNamespace() if props is None else props

    @classmethod
    def csv_columns(cls):
        return [
            ("CELL_LAYER::ID", lambda x: x.id),
            ("CELL::ID", lambda x: x.cell.id),
            ("LAYER::CHANNEL_NAME", lambda x: x.layer.channel.name),
            ("LAYER::INDEX", lambda x: x.layer.index),
            ("FOV::DIR", lambda x: x.cell.fov.dir_str()),
            ("FOV::GROUP", lambda x: x.cell.fov.group),
            ("FOV::DATE", lambda x: x.fov.date),
        ]

    @classmethod
    def csv_column_names(cls):
        return [name for name, _ in CellLayer.csv_columns()]

    def csv_column_values(self):
        return [value(self) for _, value in CellLayer.csv_columns()]
