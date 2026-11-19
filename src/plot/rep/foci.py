from src.plot.config import DEFAULT_PLOT_CONFIG

from functools import partial

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colorizer import Colorizer
from matplotlib.figure import Figure
from pandas import DataFrame
from scipy.stats import gaussian_kde

from src.defs import Cell
from .main import plot_rep
from .points import transform_points

DEFAULT_COORDINATE_0_KEY = "FOCUS::VECTOR_FROM_CENTROID_COORD_0_UM"
DEFAULT_COORDINATE_1_KEY = "FOCUS::VECTOR_FROM_CENTROID_COORD_1_UM"


def plot_foci(
    ax,
    cells,
    rep_map,
    transforms,
    *,
    foci_df: DataFrame,
    colorizer: Colorizer,
    coordinate_0_key: str = DEFAULT_COORDINATE_0_KEY,
    coordinate_1_key: str = DEFAULT_COORDINATE_1_KEY,
    symmetrize: str = DEFAULT_PLOT_CONFIG.rep_foci_symmetrize,
):
    """Representative-cell overlay that colors foci by probability density."""
    rep_points = transform_points(
        foci_df, transforms, coordinate_0_key, coordinate_1_key, symmetrize=symmetrize
    )
    if len(rep_points):
        points = rep_points[:, [1, 0]].T
        try:
            density = gaussian_kde(points)(points)
        except (np.linalg.LinAlgError, ValueError):
            density = np.ones(points.shape[1])

        order = np.argsort(density)
        ax.scatter(
            points[0, order],
            points[1, order],
            c=density[order],
            s=5,
            edgecolor="none",
            colorizer=colorizer,
        )
    return rep_points


def plot_rep_foci(
    cells: dict[str, Cell],
    foci_df: DataFrame,
    coordinate_0_key: str = DEFAULT_COORDINATE_0_KEY,
    coordinate_1_key: str = DEFAULT_COORDINATE_1_KEY,
    cmap=sns.color_palette("plasma", as_cmap=True),
    n_strata=DEFAULT_PLOT_CONFIG.rep_foci_n_strata,
    outline_margin=0.01,
    subplot_hspace=0.08,
    colorbar_fraction=0.05,
    scalebar_size_um=DEFAULT_PLOT_CONFIG.rep_foci_scalebar_size_um,
    average_method=DEFAULT_PLOT_CONFIG.rep_foci_average_method,
    colorbar_min=DEFAULT_PLOT_CONFIG.rep_foci_colorbar_min,
    colorbar_max=DEFAULT_PLOT_CONFIG.rep_foci_colorbar_max,
    symmetrize: str = DEFAULT_PLOT_CONFIG.rep_foci_symmetrize,
) -> Figure:
    """Plot foci density on representative cells using the shared layout."""
    norm = plt.Normalize(vmin=colorbar_min, vmax=colorbar_max)
    colorizer = Colorizer(cmap=cmap, norm=norm)
    plot_fn = partial(
        plot_foci,
        foci_df=foci_df,
        colorizer=colorizer,
        coordinate_0_key=coordinate_0_key,
        coordinate_1_key=coordinate_1_key,
        symmetrize=symmetrize,
    )
    return plot_rep(
        cells,
        plot_fn=plot_fn,
        mappable=plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        colorbar_label="Probability density",
        n_strata=n_strata,
        outline_margin=outline_margin,
        subplot_hspace=subplot_hspace,
        colorbar_fraction=colorbar_fraction,
        scalebar_size_um=scalebar_size_um,
        average_method=average_method,
    )
