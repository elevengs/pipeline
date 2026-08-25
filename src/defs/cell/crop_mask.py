
import numpy as np
from typing import TYPE_CHECKING

from ..fov import BoundingBox

if TYPE_CHECKING:
    from .main import Cell

# Using a labels image, create a mask cropped to the bounding box
# of the specific label we are interested in
def cropped_cell_mask(cell: "Cell", padding=5):
    props = vars(cell.props)
    dim_0_min: int = max(int(props["bbox-0"]) - padding, 0)
    dim_1_min: int = max(int(props["bbox-1"]) - padding, 0)
    dim_0_max: int = min(int(props["bbox-2"]) + padding, cell.fov.labels.shape[0])
    dim_1_max: int = min(int(props["bbox-3"]) + padding, cell.fov.labels.shape[1])

    bounding_box = BoundingBox(np.asarray([[dim_0_min, dim_0_max], [dim_1_min, dim_1_max]]))

    cropped_labels: np.ndarray = bounding_box.crop(cell.fov.labels)

    return bounding_box, (cropped_labels == cell.label).astype(np.int32)
