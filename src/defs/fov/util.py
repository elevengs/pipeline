import pandas as pd
from skimage.measure import regionprops_table

from .main import FOV


def get_all_cell_props(
    fov: FOV
):
    """Returns a ``DataFrame`` containing cell information, including their labels and morphological information.

    Uses ``regionprops`` to measure properties of cells. 
    """
    props_list = ("label", "bbox", "centroid", "orientation", "area", "axis_major_length", "axis_minor_length")
    return pd.DataFrame(
        regionprops_table(
            fov.labels,
            properties=props_list,
            spacing=(
                fov.image_dimensions.y_microns_per_pixel,
                fov.image_dimensions.x_microns_per_pixel,
            ),
        )
    )
