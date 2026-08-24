from pathlib import Path

from src.util.misc import mkdir_p


def save_fig(fig, path_basename: Path, save_svg: bool = True):
    """Returns nothing, but saves a figure as PNG and, optionally, SVG.
    """
    mkdir_p(path_basename)
    fig.savefig(
        path_basename.with_suffix(".png"),
        transparent=True,
        dpi=600,
        bbox_inches="tight",
    )
    if save_svg:
        fig.savefig(
            path_basename.with_suffix(".svg"),
            transparent=True,
            dpi=600,
            bbox_inches="tight",
        )
