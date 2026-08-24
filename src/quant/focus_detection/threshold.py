import numpy as np

from src.defs import FOV
from .config import FocusDetectionConfig

def get_detection_threshold(
    fov: FOV,
    fluor_final: np.ndarray,
    foreground: np.ndarray,
    config: FocusDetectionConfig,
    save_intensity_histograms = None,
    save_detection_threshold_samples = None
) -> float:
    """Return the minimum intensity a focus must attain in order to be detected.
    
    Arguments:
    ``fov`` -- this field of view.
    ``layer`` -- the fluorescence layer to detect foci on.
    ``foreground`` -- an ``np.ndarray`` of the same shape as fov.labels that is ``0`` in the background and ``1``on any cell.
    ``fluor_final`` -- an ``np.ndarray`` of the same shape as the fluorescence layer of interest, representing the same layer after it has been processed.
    """

    if config.detection_threshold_method == "absolute":
        return config.absolute_detection_threshold
    elif config.detection_threshold_method == "IQR_multiple":

        # In this scenario, the foreground is empty,
        # so it doesn't matter what detection threshold we
        # choose; plus, there would be no good way to do so anyway.
        if np.max(foreground) < 1:
            return 1

        foreground_fluor = fluor_final[foreground > 0]

        median_cell_fluor = np.median(foreground_fluor)
        IQR_cell_fluor = np.percentile(foreground_fluor, 75.0) - np.percentile(foreground_fluor, 25.0)

        # Calculated by a variation on the > median + 1.5 IQR criterion for outliers
        detection_threshold = (
            median_cell_fluor + config.IQR_multiple * IQR_cell_fluor
        )  # how bright a focus needs to be to detect

        return detection_threshold
    else:
        raise Exception("unsupported detection threshold method")
