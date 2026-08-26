import os
import threading
from os import path
from pathlib import Path

FIGURE_LOCK = threading.RLock()


def mkdir_p(dest: Path):
    try:
        os.makedirs(path.dirname(dest))
    except Exception as _:
        pass


# Replace root in the source path with a new directory.
# i.e. if source is ./A/B/C, and root is ./A/B, and new is ./D, returns
# ./D/C
def replace_root_with(source: Path, root: Path, new: Path) -> Path:
    return new.joinpath(source.relative_to(root))


def csv_path_for(source: Path, root: Path, output_dir: Path) -> Path:
    """Return the CSV output path corresponding to a source image."""
    return replace_root_with(source, root, output_dir).with_suffix(".csv")
