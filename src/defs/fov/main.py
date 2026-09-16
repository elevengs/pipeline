import hashlib
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from nd2 import imread
from skimage import segmentation


def file_hash(path: Path) -> str:
    digest = None
    with open(path, mode="rb") as f:
        digest = hashlib.file_digest(f, "shake_256")
    # Appears to be a bug
    return digest.hexdigest(10)  # ty: ignore[too-many-positional-arguments] # type: ignore


def str_hash(x: str) -> str:
    return hashlib.shake_256(x.encode("utf8")).hexdigest(10)


@dataclass(frozen=True)
class Channel:
    name: str
    kind: str


@dataclass(frozen=True)
class Layer:
    channel: Channel
    index: int

    additional_focus_detection_config: Namespace | None = None

    @property
    def name(self) -> str:
        return self.channel.name

    @property
    def kind(self) -> str:
        return self.channel.kind

    @property
    def is_fluor(self) -> bool:
        return self.channel.kind == "fluor"


@dataclass(frozen=True)
class ImageDimensions:
    microns: tuple[float, float]
    pixels: tuple[int, int]

    @property
    def microns_per_pixel_by_dimension(self) -> tuple[float, float]:
        return (self.microns[0] / self.pixels[0], self.microns[1] / self.pixels[1])

    @property
    def microns_per_pixel(self) -> float:
        scales = self.microns_per_pixel_by_dimension
        if not np.isclose(scales[0], scales[1], rtol=1e-3):
            raise ValueError(
                "requires approximately square pixels, but "
                f"has scales {scales[0]} and {scales[1]} microns per pixel"
            )
        return float(np.sqrt(scales[0] * scales[1]))


@dataclass(frozen=True)
class BoundingBox:
    bounds: np.ndarray  # shape (dimensions, 2)

    def crop(self, target: np.ndarray) -> np.ndarray:
        return target[
            self.bounds[0, 0] : self.bounds[0, 1], self.bounds[1, 0] : self.bounds[1, 1]
        ]


ACCEPTED_CHANNEL_KINDS = ["phase", "fluor"]


def calculate_FOV_id(labels_hash: str, source_hash: str) -> str:
    return str_hash(f"{source_hash}_{labels_hash}")


class FOV:
    id: str

    num_layers: int

    image_dimensions: ImageDimensions

    # lazy-load these because they're huge
    _labels: np.ndarray | None
    _data: np.ndarray | None

    labels_hash: str
    source_hash: str

    layers: list[Layer]

    source_path: Path
    labels_path: Path

    root: Path
    group: str
    date: str

    def __init__(
        self,
        id: str,
        date: str,
        image_dimensions: ImageDimensions,
        layers: list[Layer],
        source_path: Path,
        labels_path: Path,
        root: Path,
        labels_hash: str,
        source_hash: str,
        group: str = "all",
    ) -> None:
        self.date = date

        self.root = root
        self.image_dimensions = image_dimensions
        self.layers = layers
        self.source_path = Path(source_path)
        self.labels_path = Path(labels_path)
        self.group = group

        self._labels = None
        self._data = None

        if not isinstance(labels_hash, str) or not isinstance(source_hash, str):
            raise ValueError("labels_hash and source_hash must be strings")

        self.labels_hash = labels_hash
        self.source_hash = source_hash
        calculated_id = calculate_FOV_id(self.labels_hash, self.source_hash)

        if not id == calculated_id:
            raise Exception(
                f"got id {id}, but expected {calculated_id} based on the source hash {self.source_hash} and labels hash {self.labels_hash}"
            )
        self.id = calculated_id

        self.num_layers = len(layers)

        for layer in layers:
            if layer.kind not in ACCEPTED_CHANNEL_KINDS:
                raise Exception(
                    f"got channel kind {layer.kind}, but that is not a valid channel kind; accepted channel kinds are any of the following: {str(ACCEPTED_CHANNEL_KINDS)}"
                )
        if sum(layer.kind == "phase" for layer in layers) != 1:
            raise Exception(
                f"got {sum(layer.kind == 'phase' for layer in layers)} phase layers, but expected exactly one phase layer"
            )

    @property
    def loaded_labels(self) -> bool:
        return self._labels is not None

    @property
    def loaded_data(self) -> bool:
        return self._data is not None

    @property
    def data(self) -> np.ndarray:
        if self._data is not None:
            return self._data
        else:
            suffix = self.source_path.suffix.lower()
            if suffix == ".nd2":
                self.load_nd2(self.source_path)
            elif suffix in {".tif", ".tiff"}:
                self.load_tif(self.source_path)
            else:
                raise Exception(
                    "FOV source path does not end with .nd2, .tif, or .tiff"
                )

        return self.data

    def load_tif(self, tif_path: Path):
        calculated_hash = file_hash(tif_path)
        if self.source_hash != calculated_hash:
            raise Exception(
                f"source hash {calculated_hash} did not match expected hash {self.source_hash}"
            )
        data = iio.imread(tif_path)
        if data.ndim == 3:
            self.num_layers = data.shape[0]
        elif data.ndim == 2:
            self.num_layers = 1
            data = data[np.newaxis, ...]
        else:
            raise Exception(
                f"got FOV data of shape {data.shape}, but that is invalid; data must be either 2- or 3-dimensional, with shape (DIM_X, DIM_Y) or (NUM_LAYERS, DIM_X, DIM_Y)"
            )
        if self.num_layers != len(self.layers):
            raise Exception(
                f"got FOV data of shape {data.shape} but expected {len(self.layers)} layers"
            )

        self._data = data
        self.source_hash = calculated_hash

        self.validate_dimensions()
        return data

    def load_nd2(self, nd2_path: Path):
        calculated_hash = file_hash(nd2_path)
        if self.source_hash != calculated_hash:
            raise Exception(
                f"source hash {calculated_hash} did not match expected hash {self.source_hash}"
            )
        data = imread(nd2_path)
        if data.ndim == 3:
            self.num_layers = data.shape[0]
        elif data.ndim == 2:
            self.num_layers = 1
            data = data[np.newaxis, ...]
        else:
            raise Exception(
                f"got FOV data of shape {data.shape}, but that is invalid; data must be either 2- or 3-dimensional, with shape (DIM_X, DIM_Y) or (NUM_LAYERS, DIM_X, DIM_Y)"
            )
        if self.num_layers != len(self.layers):
            raise Exception(
                f"got FOV data of shape {data.shape} but expected {len(self.layers)} layers"
            )

        self._data = data
        self.source_hash = calculated_hash

        self.validate_dimensions()
        return data

    def load_labels(self, labels_path: Path) -> np.ndarray:

        calculated_hash = file_hash(labels_path)

        if self.labels_hash != calculated_hash:
            raise Exception(
                f"labels hash {calculated_hash} did not match expected hash {self.labels_hash}"
            )
        labels = segmentation.clear_border(iio.imread(labels_path))

        self._labels = labels
        self.labels_hash = calculated_hash

        self.validate_dimensions()
        return labels

    @property
    def labels(self) -> np.ndarray:
        if self._labels is not None:
            return self._labels
        else:
            self.load_labels(self.labels_path)

        return self.labels

    def validate_dimensions(self) -> None:
        expected = self.image_dimensions.pixels

        def validate(actual: tuple[int, int], name: str):
            if actual != expected:
                raise Exception(
                    "metadata specifies image dimensions of "
                    f"(dimension 0: {expected[0]}, dimension 1: {expected[1]}) px, "
                    f"but {name} is (dimension 0: {actual[0]}, dimension 1: {actual[1]}) px"
                )

        if self.loaded_data:
            data_dimensions = self.data.shape[-2:]
            validate(data_dimensions, "data")
        if self.loaded_labels:
            labels_dimensions = self.labels.shape
            validate(labels_dimensions, "labels")

    # This property assumes there will only exactly one phase channel
    # This condition is checked by init
    @property
    def phase_index(self) -> int:
        for layer in self.layers:
            if layer.kind == "phase":
                return layer.index
        raise Exception("FOV.phase_index called, but no phase layer found")

    def phase_data(self) -> np.ndarray:
        return self.data[self.phase_index]

    def layer_data(self, layer: Layer) -> np.ndarray:
        return self.data[layer.index]

    def dir_str(self) -> str | None:
        """Return the source directory relative to the FOV root.

        Returns ``None`` when this FOV was loaded with an explicit metadata
        file, because its root is the source file itself.
        """
        if self.root == self.source_path:
            return None
        return str(self.source_path.parent.relative_to(self.root))
