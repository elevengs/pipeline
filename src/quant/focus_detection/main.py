from collections.abc import Sequence
import numpy as np
from skimage import feature
from types import SimpleNamespace

from .fluor_layer import FluorLayerData
from src.defs import Cell, CellLayer, Focus, Layer
from src.quant.ezmaps import locate
from .stepwise_plot import plot_detected_foci
from .config import FocusDetectionConfig

def create_circle_mask(shape, center, radius):

    grids = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt(
        (grids[0] - center[0]) ** 2 + (grids[1] - center[1]) ** 2
    )
    mask = dist_from_center <= radius
    return mask

def detect_foci(
    cell: Cell,
    layer: Layer,
    fluor_layers_data: Sequence[FluorLayerData | None],
    dilated_labels: np.ndarray,
    config: FocusDetectionConfig,
    stepwise_figure_dest = None
) -> tuple[CellLayer, list[Focus]]:
    """Returns ``(cell_layer, foci)``.

    ``cell_layer`` is a ``CellLayer`` derived from this cell and this layer (obviously).
    ``foci`` is a list of each focus in the cell on this layer.

    Arguments:
    ``fluor_layers_data`` -- list of each ``layer``'s data in order by index, except ``None`` if the layer is not fluorescence.
    ``cell`` -- the cell in which the foci we are detecting reside.
    ``layer`` -- the layer of fluorescence to detect foci in
    ``dilated_labels`` -- ``fov.labels`` expanded to capture additional foci potentially just outside a given cell.
    """

    fluor_layer_data = fluor_layers_data[layer.index]

    if fluor_layer_data is None:
        raise Exception("attempting to detect foci in cell on a non-fluorescence layer")

    crop = cell.map.bounding_box.crop

    masked_fluor = crop(fluor_layer_data.fluor_final) * (
        crop(dilated_labels) == cell.label
    ).astype(np.float32)

    blobs = feature.blob_log(
        masked_fluor,
        min_sigma=config.min_sigma,
        max_sigma=config.max_sigma,
        num_sigma=config.num_sigma,
        threshold=fluor_layer_data.detection_threshold,
        overlap=config.overlap,
    )

    num_foci = len(blobs)

    cell_layer = CellLayer(
        cell, 
        layer, 
        SimpleNamespace(
            num_foci=num_foci,
            detection_threshold = fluor_layer_data.detection_threshold,
            detection_threshold_method = config.detection_threshold_method,
            detection_threshold_formula = 
                f"{str(config.absolute_detection_threshold)}" if config.detection_threshold_method == "absolute"
                else f"median + {str(config.IQR_multiple)} · IQR"
        )
    )

    foci = []

    for i, blob in enumerate(blobs):
        blob_mask = np.zeros((masked_fluor.shape[0], masked_fluor.shape[1]))

        coordinates = blob[:2]
        radius = blob[2]  # foci coordinate and radius

        # Mask each focus to determine its area and total intensity
        blob_grid = create_circle_mask(
            masked_fluor.shape,
            center=coordinates,
            radius=2 * radius,
        )       
        blob_mask[blob_grid] = 1

        masked_focus = (
            blob_mask * masked_fluor
        )

        focus_area = np.sum((masked_focus != 0).astype(np.float32))
        focus_tot_intensity = np.sum(masked_focus)
        focus_max_intensity = np.max(masked_focus)

        cell_point = locate(cell.map, coordinates)

        foci.append(
            Focus(
                cell_layer,
                i,
                cell_point,
                SimpleNamespace(
                    dim_0_coordinate = coordinates[0],
                    dim_1_coordinate = coordinates[1],
                    radius=radius,
                    area=focus_area,
                    tot_intensity=focus_tot_intensity,
                    max_intensity=focus_max_intensity
                ),
            )
        )

    if stepwise_figure_dest is not None:
        plot_detected_foci(
            cell,
            layer,
            fluor_layer_data,
            foci,
            stepwise_figure_dest
        )

    return cell_layer, foci
