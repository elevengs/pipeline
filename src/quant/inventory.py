from pathlib import Path
from types import SimpleNamespace

import numpy as np
from skimage import segmentation
from tqdm import tqdm

from src.defs import (
    FOV,
    Cell,
    CellLayer,
    Focus,
    cell_layers_to_dataframe,
    cells_to_dataframe,
    foci_to_dataframe,
)
from src.defs.fov import get_all_cell_props
from src.util.misc import mkdir_p

from .focus_detection import FocusDetectionConfig
from .focus_detection.fluor_layer import process_fluor_layer
from .focus_detection.main import detect_foci


def inventory(
    fov, focus_detection_config: FocusDetectionConfig, stepwise_figure_dest=None
) -> tuple[dict[str, Cell], dict[str, CellLayer], dict[str, Focus]]:
    """Returns ``(cells, cell_layers, foci)``.
    Computes cell polarity.
    """

    # Get regionprops for later use
    all_cell_props = get_all_cell_props(fov)
    foreground = (fov.labels > 0).astype(np.float32)

    layer_configs = {
        layer.index: focus_detection_config.with_additional_config(
            layer.additional_focus_detection_config
        )
        for layer in fov.layers
    }

    # Process and get detection threshold for each channel
    # All adapted from PolyP_analysis_v03_optimiz.ipynb
    fluor_layers_data = []
    for layer in fov.layers:
        fluor_layers_data.append(
            process_fluor_layer(fov, foreground, layer, layer_configs[layer.index])
        )

    # Dilated labels - each label is expanded by n pixels in every direction, and pixels which
    # would be covered by two labels after dilation are assigned to the closest
    # Note - the PolyP code expands the bounding box only, but this is not a good idea; both should be expanded
    dilated_labels = {
        layer.index: segmentation.expand_labels(
            fov.labels, distance=layer_configs[layer.index].expand_by
        )
        for layer in fov.layers
        if layer.is_fluor
    }

    cells = {}
    cell_layers = {}
    foci = {}

    for _, cell_props in tqdm(
        all_cell_props.iterrows(), desc=f"inventory {fov.source_path}", unit="cell"
    ):
        cell_props = cell_props.to_dict()
        cell_label = int(cell_props["label"])

        cell = Cell(
            fov,
            cell_label,
            props=SimpleNamespace(
                **{key: value for key, value in cell_props.items() if key != "label"}
            ),
        )

        cell.props.midline_length = cell.map.midline_length

        cell_foci = []

        for layer in fov.layers:
            if not layer.is_fluor:
                continue

            cell_layer, new_foci = detect_foci(
                cell,
                layer,
                fluor_layers_data,
                dilated_labels[layer.index],
                layer_configs[layer.index],
                stepwise_figure_dest=None
                if stepwise_figure_dest is None
                else stepwise_figure_dest.joinpath(
                    f"id_{cell.id}_{layer.channel.name}.png"
                ),
            )

            cell_layers[cell_layer.id] = cell_layer
            cell_foci.extend(new_foci)

        if len(cell_foci) > 0:
            # Determine cell polarity
            mean_midline_position = np.mean(
                [focus.cell_point.midline_position_px for focus in cell_foci]
            )
            cell.props.polarity = np.sign(mean_midline_position)

        cells[cell.id] = cell
        foci.update({focus.id: focus for focus in cell_foci})

    return (cells, cell_layers, foci)


def write_inventory_CSVs(
    fov: FOV,
    cells_dest: Path,
    cell_layers_dest: Path,
    foci_dest: Path,
    focus_detection_config: FocusDetectionConfig,
    stepwise_figure_dest: Path | None = None,
):
    (cells, cell_layers, foci) = inventory(
        fov,
        focus_detection_config=focus_detection_config,
        stepwise_figure_dest=stepwise_figure_dest,
    )

    cells_df = cells_to_dataframe(cells)
    cell_layers_df = cell_layers_to_dataframe(cell_layers)
    foci_df = foci_to_dataframe(foci)

    mkdir_p(cells_dest)
    cells_df.to_csv(cells_dest, index=False)
    mkdir_p(cell_layers_dest)
    cell_layers_df.to_csv(cell_layers_dest, index=False)
    mkdir_p(foci_dest)
    foci_df.to_csv(foci_dest, index=False)
