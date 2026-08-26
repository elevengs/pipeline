from pathlib import Path

from .main import FOV, calculate_FOV_id, file_hash
from .metadata import (
    from_meta,
    image_dimensions_from_meta,
    layers_from_meta,
    load_metadata,
)
from .util import find_labels


def load_tif(tif_path: Path, metadata_source: Path) -> FOV:
    meta, fov_root = load_metadata(tif_path, metadata_source)
    image_dimensions = image_dimensions_from_meta(meta)
    group = from_meta(meta, "group", str)
    date = from_meta(meta, "date", str)
    layers = layers_from_meta(meta)

    labels_path = find_labels(tif_path)

    labels_hash = file_hash(labels_path)
    tif_hash = file_hash(tif_path)
    id = calculate_FOV_id(labels_hash, tif_hash)

    fov = FOV(
        id,
        date,
        image_dimensions,
        layers,
        tif_path,
        labels_path,
        fov_root,
        labels_hash,
        tif_hash,
        group,
    )
    return fov
