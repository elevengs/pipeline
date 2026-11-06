import numpy as np
from skimage.filters import threshold_yen
from src.defs import FOV, Cell
from .config import FocusDetectionConfig


def get_detection_threshold(
    fov: FOV,
    fluor_final: np.ndarray,
    foreground: np.ndarray,
    config: FocusDetectionConfig,
    save_intensity_histograms=None,
    save_detection_threshold_samples=None,
    *,
    cell: Cell | None = None,
) -> float:
    """Return the minimum intensity a focus must attain in order to be detected.

    Arguments:
    ``fov`` -- this field of view.
    ``cell`` -- the cell whose pixels are used when the scope is ``cell``.
    ``foreground`` -- an ``np.ndarray`` of the same shape as fov.labels that is ``0`` in the background and ``1``on any cell.
    ``fluor_final`` -- an ``np.ndarray`` of the same shape as the fluorescence layer of interest, representing the same layer after it has been processed.
    """

    if config.detection_threshold_method == "absolute":
        return config.absolute_detection_threshold

    if config.detection_threshold_scope == "cell":
        if cell is None:
            raise ValueError("cell is required for cell-scoped detection thresholds")
        intensities = fluor_final[cell.mask > 0]
    else:
        intensities = fluor_final[foreground > 0]

    if intensities.size == 0:
        return 1.0

    if config.detection_threshold_method == "IQR_multiple":
        median_cell_fluor = np.median(intensities)
        IQR_cell_fluor = np.percentile(intensities, 75.0) - np.percentile(
            intensities, 25.0
        )

        # Calculated by a variation on the > median + 1.5 IQR criterion for outliers
        detection_threshold = (
            median_cell_fluor + config.IQR_multiple * IQR_cell_fluor
        )  # how bright a focus needs to be to detect

        return detection_threshold
    elif config.detection_threshold_method == "yen":
        return threshold_yen(intensities)
    else:
        raise Exception("unsupported detection threshold method")
