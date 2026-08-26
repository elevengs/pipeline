from pathlib import Path

from nd2 import ND2File

from .main import FOV, Channel, ImageDimensions, Layer, calculate_FOV_id, file_hash
from .metadata import from_meta, layers_from_meta, load_metadata
from .util import find_labels


def layers_from_nd2(f: ND2File) -> list[Layer]:

    if f.metadata.channels is None:
        raise Exception(
            f"metadata for nd2 file at {f.path} is missing the channels attribute"
        )

    layers = []

    for channel_meta in f.metadata.channels:
        kind = None

        if "brightfield" in channel_meta.microscope.modalityFlags:
            kind = "phase"
        elif "fluorescence" in channel_meta.microscope.modalityFlags:
            kind = "fluor"

        if kind is None:
            raise Exception(
                f"metadata for nd2 file at {f.path} - channel modalityFlags contains neither 'brightfield' nor 'fluorescence'"
            )

        channel = Channel(
            name="phase" if kind == "phase" else channel_meta.channel.name, kind=kind
        )

        layer = Layer(channel=channel, index=channel_meta.channel.index)

        layers.append(layer)

    return layers


def image_dimensions_from_nd2(f: ND2File) -> ImageDimensions:

    if f.metadata.channels is None:
        raise Exception(
            f"metadata for nd2 file at {f.path} is missing the channels attribute"
        )

    dim = None
    cal = None

    for channel_meta in f.metadata.channels:
        dim_0, dim_1 = channel_meta.volume.voxelCount[:2]
        cal_0, cal_1 = channel_meta.volume.axesCalibration[:2]
        new_dim = (int(dim_0), int(dim_1))
        new_cal = (float(cal_0), float(cal_1))

        if dim is None:
            dim = new_dim
            cal = new_cal
        elif dim != new_dim:
            raise Exception(
                f"metadata for nd2 file at {f.path} has different dimensions for different channels; this is not permitted"
            )
        elif cal != new_cal:
            raise Exception(
                f"metadata for nd2 file at {f.path} has different calibration ratios for different channels; this is not permitted"
            )

    if dim is None or cal is None:
        raise Exception(
            f"metadata for nd2 file at {f.path} has insufficient channelwise dimension data"
        )

    return ImageDimensions(
        pixels=(dim[1], dim[0]), microns=(dim[1] * cal[1], dim[0] * cal[0])
    )


def date_from_nd2(f: ND2File) -> str:
    timestamp = f.text_info["date"]  # type: ignore

    return timestamp.split(sep=" ")[0]


def load_nd2(nd2_path: Path, metadata_source: Path) -> FOV:

    meta, fov_root = load_metadata(nd2_path, metadata_source)
    group = from_meta(meta, "group", str)
    labels_path = find_labels(nd2_path)
    meta_layers = layers_from_meta(meta)

    labels_hash = file_hash(labels_path)
    nd2_hash = file_hash(nd2_path)
    id = calculate_FOV_id(labels_hash, nd2_hash)

    with ND2File(nd2_path) as ndfile:
        image_dimensions = image_dimensions_from_nd2(ndfile)
        date = date_from_nd2(ndfile)
        layers = layers_from_nd2(ndfile)

        for layer, meta_layer in zip(layers, meta_layers):
            if (
                layer.channel.kind != meta_layer.channel.kind
                or layer.index != meta_layer.index
            ):
                raise Exception(
                    f"nd2 at {nd2_path} has channel data which conflicts with meta.json for channel {layer.index}"
                )

        fov = FOV(
            id,
            date,
            image_dimensions,
            meta_layers,
            nd2_path,
            labels_path,
            fov_root,
            labels_hash,
            nd2_hash,
            group,
        )

    return fov
