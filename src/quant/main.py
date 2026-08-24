import argparse
from pathlib import Path

from .batch_inventory import batch_inventory
from .focus_detection import FocusDetectionConfig, add_detection_config_args


def main() -> None:
    parser = argparse.ArgumentParser(prog="batch_inventory", usage="%(prog)s [options]")

    parser.add_argument(
        "-i",
        "--include",
        nargs="+",
        action="extend",
        default=None,
        help=(
            "Include source ND2 or TIFF files using one or more glob patterns. "
            "The option may be passed multiple times; default: ['**/*.nd2']."
        ),
    )
    parser.add_argument(
        "-e",
        "--exclude",
        nargs="+",
        default=[],
        help="Exclude files or directories using one or more glob patterns; default: [].",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory against which include and exclude patterns are evaluated; default: '.'.",
    )

    add_detection_config_args(parser)

    parser.add_argument(
        "--save-stepwise-figures",
        type=Path,
        help=(
            "Directory for per-cell stepwise figures. The source directory "
            "structure relative to --root is preserved."
        ),
    )
    parser.add_argument(
        "cells_dir",
        type=Path,
        help="Directory for per-FOV cell CSV files.",
    )
    parser.add_argument(
        "cell_layers_dir",
        type=Path,
        help="Directory for per-FOV cell-layer CSV files.",
    )
    parser.add_argument(
        "foci_dir",
        type=Path,
        help="Directory for per-FOV focus CSV files.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Number of worker threads; set to 0 to use the main thread. Default: 5.",
    )

    args = parser.parse_args()
    include = ["**/*.nd2"] if args.include is None else args.include

    batch_inventory(
        root=args.root,
        include=include,
        exclude=args.exclude,
        focus_detection_config=FocusDetectionConfig.from_args(args),
        cells_dir=args.cells_dir,
        cell_layers_dir=args.cell_layers_dir,
        foci_dir=args.foci_dir,
        stepwise_figures_dir=args.save_stepwise_figures,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
