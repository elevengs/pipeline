from pathlib import Path

from nd2 import ND2File

from .main import Channel, FOV, ImageDimensions, Layer, calculate_FOV_id, file_hash
from .metadata import from_meta, layers_from_meta, load_metadata

def layers_from_nd2(f: ND2File) -> list[Layer]:

    if f.metadata.channels is None:
        raise Exception(f"metadata for nd2 file at {f.path} is missing the channels attribute")

    layers = []

    for channel_meta in f.metadata.channels:
        kind = None

        if "brightfield" in channel_meta.microscope.modalityFlags:
            kind = "phase"
        elif "fluorescence" in channel_meta.microscope.modalityFlags:
            kind = "fluor"
            
        if kind is None:
            raise Exception(f"metadata for nd2 file at {f.path} - channel modalityFlags contains neither 'brightfield' nor 'fluorescence'")
        
        channel = Channel(
            name = "phase" if kind == "phase" else channel_meta.channel.name,
            kind = kind
        )

        layer = Layer(
            channel = channel,
            index = channel_meta.channel.index
        )

        layers.append(layer)

    return layers
    

def image_dimensions_from_nd2(f: ND2File) -> ImageDimensions:

    if f.metadata.channels is None:
        raise Exception(f"metadata for nd2 file at {f.path} is missing the channels attribute")

    res = None

    for channel_meta in f.metadata.channels:
        new_row_dim, new_col_dim, _ = channel_meta.volume.voxelCount
        new_row_ax_cal, new_col_ax_cal, _ = channel_meta.volume.axesCalibration

        new_res: dict[str, int | float]= {
            "row_dim": new_row_dim,
            "col_dim": new_col_dim,
            "row_ax_cal": new_row_ax_cal,
            "col_ax_cal": new_col_ax_cal
        }

        if res is None:
            res = new_res
        elif res != new_res:
            raise Exception(f"metadata for nd2 file at {f.path} has different dimensions for different channels; this is not permitted")
    
    if res is None:
        raise Exception(f"metadata for nd2 file at {f.path} has no channelwise dimension data")

    if res["row_ax_cal"] != res["col_ax_cal"]:
        raise Exception(f"row and column calibration values differ in a channel in nd2 file at {f.path}; this is not allowed")

    rows = int(res["row_dim"])
    cols = int(res["col_dim"])
    row_ax_cal = res["row_ax_cal"]
    col_ax_cal = res["col_ax_cal"]

    return ImageDimensions(
        x_microns = rows * row_ax_cal,
        y_microns = cols * col_ax_cal,
        x_pixels = rows,
        y_pixels = cols
    )

def date_from_nd2(f: ND2File) -> str:
    timestamp = f.text_info["date"]

    return timestamp.split(sep = " ")[0]

def load_nd2(nd2_path: Path, metadata_source: Path) -> FOV:

    meta, fov_root = load_metadata(nd2_path, metadata_source)
    group = from_meta(meta, "group", str)
    labels_path = nd2_path.with_name(f"{nd2_path.stem}_labels.png")
    meta_layers = layers_from_meta(meta)

    labels_hash = file_hash(labels_path)
    nd2_hash = file_hash(nd2_path)
    id = calculate_FOV_id(labels_hash, nd2_hash)

    with ND2File(nd2_path) as ndfile:
        image_dimensions = image_dimensions_from_nd2(ndfile)
        date = date_from_nd2(ndfile)
        layers = layers_from_nd2(ndfile)

        for layer, meta_layer in zip(layers, meta_layers):
            if layer.channel.kind != meta_layer.channel.kind or layer.index != meta_layer.index:
                raise Exception(f"nd2 at {nd2_path} has channel data which conflicts with meta.json for channel {layer.index}")

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
            group
        )

    return fov
    
