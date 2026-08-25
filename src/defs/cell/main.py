from types import SimpleNamespace

import numpy as np

from ..fov import FOV
from src.quant.ezmaps import CellMap

class Cell:

    id: str
    label: int

    fov: FOV
    props: SimpleNamespace

    def __init__(
        self,
        fov: FOV,
        label: int,
        id: str | None = None,
        props: SimpleNamespace | None = None,
    ):
        calculated_id = f"{fov.id}_{label}"

        if id is not None:
            self.id = id
            if id != calculated_id:
                raise Exception(
                    f"calculated cell id {calculated_id}, but expected id {id}"
                )
        self.id = calculated_id

        self.label = label
        self.fov = fov
        self.props = SimpleNamespace() if props is None else props
        self._map = None

    def csv_columns():
        return [
            ("CELL::ID", lambda x: x.id),
            ("CELL::LABEL", lambda x: x.label),
            ("FOV::DIR", lambda x: x.fov.dir_str()),
            ("FOV::GROUP", lambda x: x.fov.group),
            ("FOV::DATE", lambda x: x.fov.date)
        ]

    def csv_column_names():
        return [name for name, _ in Cell.csv_columns()]

    def csv_column_values(self):
        return [value(self) for _, value in Cell.csv_columns()]

    @property
    def mask(self) -> np.ndarray:
        if self.fov.labels is None:
            raise ValueError("cell mask is unavailable because FOV labels are not loaded")
        return (self.fov.labels == self.label).astype(np.int32)


    # lazy-load this because it depends on fov.labels
    _map: CellMap | None

    @property
    def map(self) -> CellMap:
        if self._map is None:
            from .crop_mask import cropped_cell_mask

            bounding_box, mask = cropped_cell_mask(self)

            self._map = CellMap(bounding_box, mask)
        return self._map
        
