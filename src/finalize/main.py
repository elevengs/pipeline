import argparse
from pathlib import Path

from tqdm.auto import tqdm

from src.util.incl_excl import select_files

from .finalize_group import group_output_dir, make_group_outputs, sources_by_group


def finalize(
    root: Path,
    include: list[str],
    exclude: list[str],
    out: Path,
    cells_dir: Path,
    cell_layers_dir: Path,
    foci_dir: Path,
    min_bio_reps: int,
    max_workers: int,
    enable_spideymaps: bool,
) -> None:
    sources = list(select_files(root, include=include, exclude=exclude))
    for group, group_sources in tqdm(
        sources_by_group(sources, root).items(),
        desc="Handling groups",
        unit="group",
    ):
        make_group_outputs(
            group_sources,
            root,
            cells_dir,
            cell_layers_dir,
            foci_dir,
            group_output_dir(out, group),
            min_bio_reps,
            max_workers,
            enable_spideymaps,
        )


def main() -> None:
    parser = argparse.ArgumentParser(prog="finalize", usage="%(prog)s [options]")
    parser.add_argument(
        "-i",
        "--include",
        nargs="+",
        action="extend",
        default=None,
        help="Include source ND2 or TIFF files; default: ['**/*.nd2'].",
    )
    parser.add_argument(
        "-e",
        "--exclude",
        nargs="+",
        default=[],
        help="Exclude files or directories using glob patterns; default: [].",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory against which include, exclude, and output paths are resolved.",
    )
    parser.add_argument(
        "out",
        type=Path,
        help="Directory where finalized outputs will be written.",
    )
    parser.add_argument(
        "--enable-spideymaps",
        action="store_true",
        help="Also generate Spideymaps atlas data and plots under OUT/spideymaps.",
    )
    parser.add_argument(
        "--cells-dir",
        type=Path,
        required=True,
        help="Directory containing cell CSVs written by quant.",
    )
    parser.add_argument(
        "--cell-layers-dir",
        type=Path,
        required=True,
        help="Directory containing cell-layer CSVs written by quant.",
    )
    parser.add_argument(
        "--foci-dir",
        type=Path,
        required=True,
        help="Directory containing foci CSVs written by quant.",
    )
    parser.add_argument(
        "--min-bio-reps",
        type=int,
        default=3,
        help="Minimum biological replicates per group; default: 3.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Number of worker threads; set to 0 to use the main thread.",
    )

    args = parser.parse_args()
    finalize(
        root=args.root,
        include=["**/*.nd2"] if args.include is None else args.include,
        exclude=args.exclude,
        out=args.out,
        cells_dir=args.cells_dir,
        cell_layers_dir=args.cell_layers_dir,
        foci_dir=args.foci_dir,
        min_bio_reps=args.min_bio_reps,
        max_workers=args.max_workers,
        enable_spideymaps=args.enable_spideymaps,
    )


if __name__ == "__main__":
    main()
