from typing import Any
import json
from pathlib import Path

from .main import Channel, ImageDimensions, Layer

def _read_meta_file(meta_path: Path) -> dict:
    with meta_path.open("r") as file:
        meta = json.load(file)

    if not isinstance(meta, dict):
        raise ValueError(f"metadata in {meta_path} must be a dictionary")
    return meta


def load_meta_from_dir(directory: Path) -> dict | None:
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        return None
    return _read_meta_file(meta_path)


def load_metadata(
    source_path: Path,
    metadata_source: Path,
) -> tuple[dict, Path]:
    """Load metadata and determine the root associated with an FOV.

    A directory source loads and merges ``meta.json`` files from that
    directory through the source directory. A specific ``meta.json`` file is
    used exactly as provided, and the source file path itself becomes the FOV root.
    """
    source_path = Path(source_path)
    metadata_source = Path(metadata_source)
    resolved_source_path = source_path.resolve()
    resolved_metadata_source = metadata_source.resolve()

    if resolved_metadata_source.is_file():
        if metadata_source.name != "meta.json":
            raise ValueError("metadata_source must be a directory or a meta.json file")
        return _read_meta_file(resolved_metadata_source), source_path

    if not resolved_metadata_source.is_dir():
        raise FileNotFoundError(metadata_source)

    try:
        resolved_source_path.relative_to(resolved_metadata_source)
    except ValueError as exc:
        raise ValueError(
            f"{source_path} is not inside metadata root {metadata_source}"
        ) from exc

    directories = []
    current_dir = resolved_source_path.parent
    while True:
        directories.append(current_dir)
        if current_dir == resolved_metadata_source:
            break
        current_dir = current_dir.parent

    metadata = {}
    found_metadata = False
    for directory in reversed(directories):
        directory_metadata = load_meta_from_dir(directory)
        if directory_metadata is not None:
            metadata.update(directory_metadata)
            found_metadata = True

    if not found_metadata:
        raise FileNotFoundError(
            f"did not find meta.json for {source_path} in {metadata_source}"
        )

    return metadata, metadata_source


def from_meta(meta: dict, key: str, value_type: type):
    if key not in meta:
        raise ValueError(f'metadata {meta} missing key "{key}"')

    value = meta[key]
    if isinstance(value, value_type):
        return value

    raise ValueError(
        f'metadata {meta} must have key "{key}" with value of type {value_type}'
    )


def layer_from_dict(layer_dict: dict, index: int) -> Layer:
    if not isinstance(layer_dict, dict):
        raise ValueError("layer information must be a dictionary")

    name = layer_dict.get("name")
    kind = layer_dict.get("kind")
    if not isinstance(name, str):
        raise ValueError('channel information must include a string "name"')
    if not isinstance(kind, str):
        raise ValueError('channel information must include a string "kind"')

    return Layer(Channel(name, kind), index)


def layers_from_meta(meta: dict) -> list[Layer]:
    layer_dicts = from_meta(meta, "layers", list)
    return [layer_from_dict(layer_dict, i) for i, layer_dict in enumerate(layer_dicts)]


def image_dimensions_from_meta(meta: dict) -> ImageDimensions:
    dimensions = from_meta(meta, "image_dimensions", dict)

    values: dict[str, Any] = {}
    for axis in ("x", "y"):
        axis_dimensions = from_meta(dimensions, axis, dict)
        microns = axis_dimensions.get("microns")
        pixels = axis_dimensions.get("pixels")

        if isinstance(microns, bool) or not isinstance(microns, (int, float)) or microns <= 0:
            raise ValueError(f"metadata image_dimensions.{axis}.microns must be positive")
        if isinstance(pixels, bool) or not isinstance(pixels, int) or pixels <= 0:
            raise ValueError(f"metadata image_dimensions.{axis}.pixels must be positive")

        values[f"{axis}_microns"] = float(microns)
        values[f"{axis}_pixels"] = int(pixels)

    return ImageDimensions(**values)
