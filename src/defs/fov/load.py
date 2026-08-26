from pathlib import Path

from .load_nd2 import load_nd2
from .load_tif import load_tif
from .main import FOV


def load(
    source_path: Path,
    metadata_source: Path,
) -> FOV:
    """Load a TIFF or ND2 source as an FOV.

    ``metadata_source`` may be either a directory containing inherited
    ``meta.json`` files or a specific ``meta.json`` file.
    """
    source_path = Path(source_path)
    suffix = source_path.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        return load_tif(
            source_path,
            metadata_source,
        )
    if suffix == ".nd2":
        return load_nd2(
            source_path,
            metadata_source,
        )

    raise ValueError(
        f"unsupported FOV source extension {suffix!r}; expected .nd2, .tif, or .tiff"
    )
