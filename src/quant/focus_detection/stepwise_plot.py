from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from src.defs import Cell, Focus, Layer
from src.quant.focus_detection.fluor_layer import FluorLayerData
from src.util.misc import FIGURE_LOCK
from src.util.save import save_fig


def plot_detected_foci(
    cell: Cell,
    layer: Layer,
    fluor_layer_data: FluorLayerData,
    foci: list[Focus],
    dest: Path,
):
    fov = cell.fov
    crop = cell.map.bounding_box.crop

    thresholded_fluor_final = fluor_layer_data.fluor_final * (
        fluor_layer_data.fluor_final > fluor_layer_data.detection_threshold
    ).astype(np.float32)

    plot_titles_and_contents = [
        ("original", fov.layer_data(layer)),
        ("background corrected", fluor_layer_data.fluor_minus_bg),
        ("denoised", fluor_layer_data.fluor_minus_bg_sharp),
        ("final", fluor_layer_data.fluor_final),
        ("thresholded", thresholded_fluor_final),
        ("map", cell.mask),
        ("phase", fov.phase_data()),
    ]

    # Show the detected blobs on the fluor_final image,
    # because that is what blob_log is called with
    SHOW_BLOBS_ON_IDX = 3

    # Show the mapped foci on the "map" layer
    SHOW_MAP_ON_IDX = 5

    with FIGURE_LOCK:
        fig, unraveled_axes = plt.subplots(
            2, 4, figsize=(15, 10), sharex=False, sharey=False
        )

        axes = unraveled_axes.ravel()
        for axis in axes[len(plot_titles_and_contents) :]:
            axis.set_visible(False)

        def draw_cell_boundary(i):
            axes[i].contour(crop(cell.mask), levels=1, colors="w")

        def show(i, x):
            axes[i].imshow(crop(x))

        for _, focus in enumerate(foci):
            center = (focus.props.dim_1_coordinate, focus.props.dim_0_coordinate)
            c = plt.Circle(  # type: ignore (bug)
                center, focus.props.radius, linewidth=2, fill=False, color="red"
            )
            axes[SHOW_BLOBS_ON_IDX].add_patch(c)

        cell.map.show(
            ax=axes[SHOW_MAP_ON_IDX],
            render=False,
            cell_points=[focus.cell_point for focus in foci],
        )

        for i, (axis, (plot_title, plot_contents)) in enumerate(
            zip(axes, plot_titles_and_contents)
        ):
            axis.set_title(plot_title)
            show(i, plot_contents)
            if i == SHOW_MAP_ON_IDX:
                continue
            draw_cell_boundary(i)

        plt.tight_layout()

        save_fig(fig, path_basename=dest, save_svg=False)
        plt.close(fig)
