from pathlib import Path

import pandas as pd
from skimage.measure import regionprops_table

from .main import FOV


def find_labels(source_path: Path) -> Path:
    """Find the label image associated with an FOV source.

    Label images are stored beside their source using the source stem and a
    ``_labels`` suffix. PNG and TIFF label images are supported; PNG takes
    precedence when both are present.
    """
    png_candidate = source_path.with_name(f"{source_path.stem}_labels.png")
    tif_candidate = source_path.with_name(f"{source_path.stem}_labels.tif")
    tiff_candidate = source_path.with_name(f"{source_path.stem}_labels.tiff")

    if png_candidate.is_file():
        return png_candidate
    if tif_candidate.is_file():
        return tif_candidate
    if tiff_candidate.is_file():
        return tiff_candidate
    raise FileNotFoundError(
        f"no label image found for source {source_path}; expected one of "
        f"{png_candidate.name}, {tif_candidate.name}, or {tiff_candidate.name}"
    )


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
            spacing=fov.image_dimensions.microns_per_pixel_by_dimension
        )
    )
