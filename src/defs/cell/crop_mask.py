
import numpy as np
from typing import TYPE_CHECKING

from ..fov import Region

if TYPE_CHECKING:
    from .main import Cell

# Using a labels image, create a mask cropped to the region
# of the specific label we are interested in
def cropped_cell_mask(cell: "Cell", padding=5):
    props = vars(cell.props)
    rowmin = max(int(props["bbox-0"]) - padding, 0)
    colmin = max(int(props["bbox-1"]) - padding, 0)
    rowmax = min(int(props["bbox-2"]) + padding, cell.fov.labels.shape[0])
    colmax = min(int(props["bbox-3"]) + padding, cell.fov.labels.shape[1])

    region = Region(rowmin, colmin, rowmax, colmax)

    cropped_labels = cell.fov.labels[rowmin:rowmax, colmin:colmax]

    return region, (cropped_labels == cell.label).astype(np.int32)
