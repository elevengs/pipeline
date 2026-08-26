from types import SimpleNamespace

import numpy as np

from src.quant.ezmaps import CellPoint

from ..cell_layer import CellLayer


class Focus:
    id: str
    index: int
    cell_layer: CellLayer
    cell_point: CellPoint
    props: SimpleNamespace

    def __init__(
        self,
        cell_layer: CellLayer,
        index: int,
        cell_point: CellPoint,
        props: SimpleNamespace | None = None,
    ):
        self.id = f"{cell_layer.id}_{index}"
        self.cell_layer = cell_layer
        self.index = index
        self.cell_point = cell_point
        self.props = SimpleNamespace() if props is None else props

    @property
    def cell(self):
        return self.cell_layer.cell

    @property
    def fov(self):
        return self.cell_layer.cell.fov

    @property
    def channel(self):
        return self.cell_layer.layer.channel

    @property
    def layer(self):
        return self.cell_layer.layer

    @property
    def midline_position_um(self):
        return (
            self.cell_point.midline_position_px
            * self.fov.image_dimensions.microns_per_pixel
        )

    @property
    def midline_position_pol_um(self):
        use_polarity = 1

        if (
            getattr(self.cell.props, "polarity", None) is not None
            and self.cell.props.polarity < 0
        ):
            use_polarity = -1
        return self.midline_position_um * use_polarity

    @property
    def offset_from_midline_um(self):
        return (
            self.cell_point.offset_from_midline_px
            * self.fov.image_dimensions.microns_per_pixel
        )

    @property
    def cell_midline_length_um(self):
        return (
            self.cell.map.midline_length * self.fov.image_dimensions.microns_per_pixel
        )

    @classmethod
    def csv_columns(cls):
        return [
            ("FOCUS::ID", lambda x: x.id),
            ("FOCUS::INDEX", lambda x: x.index),
            ("CELL_LAYER::ID", lambda x: x.cell_layer.id),
            ("CELL::ID", lambda x: x.cell.id),
            ("FOV::DIR", lambda x: x.cell.fov.dir_str()),
            ("FOV::GROUP", lambda x: x.cell.fov.group),
            ("FOV::DATE", lambda x: x.fov.date),
            (
                "FOV::MICRONS_PER_PIXEL",
                lambda x: x.fov.image_dimensions.microns_per_pixel,
            ),
            (
                "CELL_LAYER::PROPS::NUM_FOCI",
                lambda x: getattr(x.cell_layer.props, "num_foci", np.nan),
            ),
            ("LAYER::INDEX", lambda x: x.layer.index),
            ("LAYER::CHANNEL_NAME", lambda x: x.channel.name),
            ("FOCUS::MIDLINE_POSITION_PX", lambda x: x.cell_point.midline_position_px),
            (
                "FOCUS::OFFSET_FROM_MIDLINE_PX",
                lambda x: x.cell_point.offset_from_midline_px,
            ),
            ("FOCUS::MIDLINE_POSITION_UM", lambda x: x.midline_position_um),
            ("FOCUS::MIDLINE_POSITION_POL_UM", lambda x: x.midline_position_pol_um),
            ("FOCUS::OFFSET_FROM_MIDLINE_UM", lambda x: x.offset_from_midline_um),
            ("CELL::MIDLINE_LENGTH_UM", lambda x: x.cell_midline_length_um),
        ]

    @classmethod
    def csv_column_names(cls):
        return [name for name, _ in Focus.csv_columns()]

    def csv_column_values(self):
        return [value(self) for _, value in Focus.csv_columns()]
