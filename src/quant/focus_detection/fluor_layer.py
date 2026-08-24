import numpy as np

from skimage import filters, restoration
from src.defs import FOV, Layer
from .threshold import get_detection_threshold
from .config import FocusDetectionConfig

class FluorLayerData:
    fluor_minus_bg: np.ndarray
    fluor_minus_bg_sharp: np.ndarray
    fluor_final: np.ndarray
    quantiles: np.ndarray
    detection_threshold: float

    def __init__(
        self,
        fluor_minus_bg: np.ndarray,
        fluor_minus_bg_sharp: np.ndarray,
        fluor_final: np.ndarray,
        detection_threshold: float
    ):
        self.fluor_minus_bg = fluor_minus_bg
        self.fluor_minus_bg_sharp = fluor_minus_bg_sharp
        self.fluor_final = fluor_final
        self.detection_threshold = detection_threshold
        

def process_fluor_layer(
    fov: FOV,
    foreground: np.ndarray,
    layer: Layer,
    config: FocusDetectionConfig
) -> FluorLayerData | None:
    """Returns ``FluorLayerData`` for this ``layer``.

    Arguments:
    ``fov`` -- this field of view.
    ``foreground`` -- an ``np.ndarray`` of the same dimensions as ``fov.labels``, which is ``1`` whenever ``fov.labels > 0`` and ``0`` everywhere else.
    ``layer`` -- the Layer to process.
    """

    if not layer.is_fluor:
        return None

    i = layer.index

    fluor_minus_bg = fov.data[i] - restoration.rolling_ball(
        fov.data[i],
        radius=config.bg_sub_radius
    )

    fluor_minus_bg_sharp = filters.unsharp_mask(
        fluor_minus_bg,
        radius=config.denoise_radius,
        amount=config.denoise_amount,
        preserve_range=True,
    )

    fluor_final = filters.gaussian(
        fluor_minus_bg_sharp,
        config.gaussian_sigma
    )


    detection_threshold = get_detection_threshold(
        fov, fluor_final, foreground, config
    )

    return FluorLayerData(
       fluor_minus_bg,
       fluor_minus_bg_sharp,
       fluor_final,
       detection_threshold 
    ) 
