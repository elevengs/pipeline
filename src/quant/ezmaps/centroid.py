import numpy as np
from skimage.measure import regionprops


def centroid(mask: np.ndarray) -> np.ndarray:
    """Returns [row, col] position of the cell centroid."""
    return regionprops(mask)[0].centroid
