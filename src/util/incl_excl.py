from pathlib import Path
from collections.abc import Iterable, Iterator
from pathspec import GitIgnoreSpec


def select_files(
    root: str | Path,
    include: Iterable[str] = ("**/*.nd2",),
    exclude: Iterable[str] = (),
) -> Iterator[Path]:
    """Yield files under root that match include patterns and do not match exclude patterns.

    Patterns use gitignore-style syntax and are evaluated relative to root.
    """
    root = Path(root)

    include_spec = GitIgnoreSpec.from_lines(include)
    exclude_spec = GitIgnoreSpec.from_lines(exclude)

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        rel = path.relative_to(root).as_posix()

        if include_spec.match_file(rel) and not exclude_spec.match_file(rel):
            yield path
