from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def handle_sources_with(
    sources: Iterable[Path],
    fn: Callable[[Path], object],
    max_workers: int,
) -> Iterator[tuple[Path, object]]:
    """Process ``sources`` using ``fn``, using either a thread pool if ``max_workers > 0`` or a simple loop otherwise.
    """
    sources = list(sources)
    if max_workers == 0:
        for source in sources:
            try:
                result = fn(source)
            except Exception as exc:
                yield source, exc
            else:
                yield source, result
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_source = {executor.submit(fn, source): source for source in sources}
        for future in as_completed(future_to_source):
            source = future_to_source[future]
            try:
                result = future.result()
            except Exception as exc:
                yield source, exc
            else:
                yield source, result
