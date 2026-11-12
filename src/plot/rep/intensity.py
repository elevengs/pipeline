"""Fluorescence intensity overlays from masked FOV pixels."""

from collections.abc import Callable
from functools import partial

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy.spatial.distance import cdist

from src.defs import Cell
from .main import plot_rep


IntensitySmoother = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

def gaussian_intensity(
    points, 
    values, 
    query_points, 
    *, 
    sigma_um=0.1,
    query_size = 128,
    answer_size = 8192
):
    """Returns the smoothed intensity at a series of points.

    ``points`` are the point coordinates.
    ``values`` are the intensity values for `points`, so they should be of the same shape.
    ``query_points`` are the coordinates of the output smoothed intensities.
    ``sigma_um`` is the standard deviation (in micrometers) of the Gaussian kernel used for smoothing.
    ``query_size`` is the number of points queried in one batch.
    ``answer_size`` is the number of `values` used in answering one query. 
    """
    if not np.isfinite(sigma_um) or sigma_um <= 0:
        raise ValueError("Kernel standard deviation must be finite and positive")
    result = np.full(len(query_points), np.nan)
    for start in range(0, len(query_points), query_size):
        query = query_points[start : start + query_size]
        numerator = np.zeros(len(query))
        denominator = np.zeros(len(query))
        for sample_start in range(0, len(points), answer_size):
            samples = points[sample_start : sample_start + answer_size]
            weights = cdist(query, samples, metric="sqeuclidean")
            weights /= -2 * sigma_um**2
            np.exp(weights, out=weights)
            numerator += weights @ values[sample_start : sample_start + answer_size]
            denominator += weights.sum(axis=1)
        np.divide(
            numerator,
            denominator,
            out=result[start : start + len(query)],
            where=denominator > 0,
        )
    return result


def plot_intensity(
    ax,
    cells,
    rep_map,
    transforms,
    *,
    channel_name: str,
    smooth_fn: IntensitySmoother,
    grid_spacing_um: float,
    cmap,
    norm,
    symmetrize = "xy"
):
    """Evaluate continuous-position intensity smoothing and draw an overlay.

    ``smooth_fn(points, values, query_points)`` returns one intensity per query.
    Positions are (row, column) vectors in microns. The evaluation grid only
    controls display resolution; input sample positions are never quantized.
    """
    boundary = rep_map.boundary_by_arc_length(np.linspace(0, rep_map.perimeter, 500))
    coordinates = [
        np.linspace(
            np.min(boundary[:, i]),
            np.max(boundary[:, i]),
            max(2, int(np.ceil(np.ptp(boundary[:, i]) / grid_spacing_um)) + 1),
        )
        for i in range(2)
    ]
    point_sets = []
    value_sets = []
    for cell in cells:
        if cell.id not in transforms:
            continue
        layers = [
            layer
            for layer in cell.fov.layers
            if layer.name == channel_name and layer.is_fluor
        ]
        if len(layers) != 1:
            raise ValueError(
                f'expected one fluorescence layer "{channel_name}" for cell {cell.id}'
            )
        cell_map = cell.map
        crop = cell_map.bounding_box.crop(cell.fov.layer_data(layers[0]))
        mask = np.asarray(cell_map.mask, dtype=bool) & np.isfinite(crop)
        values = np.asarray(crop[mask], dtype=float)
        vectors = (
            np.argwhere(mask) - cell_map.centroid
        ) * cell.fov.image_dimensions.microns_per_pixel
        points = vectors @ transforms[cell.id].T
        point_sets.append(points)
        value_sets.append(values)
        if "x" in symmetrize:
            mirrored_x = points.copy()
            mirrored_x[:, 0] = (mirrored_x[:, 0] * -1)
            point_sets.append(mirrored_x)
            value_sets.append(values)
        if "y" in symmetrize:
            mirrored_y = points.copy()
            mirrored_y[:, 1] = (mirrored_y[:, 1] * -1)
            point_sets.append(mirrored_y)
            value_sets.append(values)

    rows, cols = np.meshgrid(*coordinates, indexing="ij")
    queries = np.column_stack([rows.ravel(), cols.ravel()])
    points = np.concatenate(point_sets) if point_sets else np.empty((0, 2))
    values = np.concatenate(value_sets) if value_sets else np.empty(0)
    intensity = np.full(len(queries), np.nan)
    if len(points):
        intensity = np.asarray(smooth_fn(points, values, queries), dtype=float)
        if intensity.shape != (len(queries),):
            raise ValueError("smooth_fn must return one intensity per query point")
    intensity = intensity.reshape(rows.shape)
    outline = Path(boundary[:, [1, 0]])
    # Evaluate beyond the outline too, then clip the interpolated mesh exactly
    # to the boundary, avoiding gaps caused by masking boundary grid vertices.
    artist = ax.pcolormesh(
        coordinates[1],
        coordinates[0],
        np.ma.masked_invalid(intensity),
        cmap=cmap,
        norm=norm,
        shading="gouraud",
        rasterized=True,
    )
    artist.set_clip_path(PathPatch(outline, transform=ax.transData))
    return artist


def plot_rep_intensity(
    cells: dict[str, Cell],
    channel_name: str,
    *,
    smooth_fn: IntensitySmoother | None = None,
    smoothing_sigma_um: float = 0.1,
    grid_spacing_um: float = 0.05,
    cmap="plasma",
    norm: Normalize | None = None,
    n_strata=4,
    outline_margin=0.01,
    subplot_hspace=0.08,
    colorbar_fraction=0.05,
):
    """Plot raw fluorescence intensity averaged over aligned cell pixels.

    Uses each cell's FOV fluorescence channel, cropped to its map's mask. 
    No background subtraction or per-cell normalization is done.
    """
    if not np.isfinite(grid_spacing_um) or grid_spacing_um <= 0:
        raise ValueError("grid_spacing_um must be finite and positive")
    if not np.isfinite(smoothing_sigma_um) or smoothing_sigma_um <= 0:
        raise ValueError("smoothing_sigma_um must be finite and positive")
    if smooth_fn is None:
        smooth_fn = partial(gaussian_intensity, sigma_um=smoothing_sigma_um)
    shared_norm = Normalize() if norm is None else norm
    # Set limits after every stratum has been rendered, avoiding autoscaling
    # the shared norm to only the first stratum.
    vmin, vmax = shared_norm.vmin, shared_norm.vmax
    shared_norm.vmin = 0 if vmin is None else vmin
    shared_norm.vmax = 1 if vmax is None else vmax
    artists = []

    def overlay(ax, stratum, rep_map, transforms):
        artists.append(
            plot_intensity(
                ax,
                stratum,
                rep_map,
                transforms,
                channel_name=channel_name,
                smooth_fn=smooth_fn,
                grid_spacing_um=grid_spacing_um,
                cmap=cmap,
                norm=shared_norm,
            )
        )

    fig = plot_rep(
        cells,
        overlay,
        mappable=plt.cm.ScalarMappable(norm=shared_norm, cmap=cmap),
        colorbar_label="Fluorescence intensity",
        n_strata=n_strata,
        outline_margin=outline_margin,
        subplot_hspace=subplot_hspace,
        colorbar_fraction=colorbar_fraction,
    )
    finite_values = [artist.get_array().compressed() for artist in artists]
    values = np.concatenate(finite_values)
    if len(values):
        # Update both limits together before notifying the colorbar.
        with shared_norm.callbacks.blocked():
            shared_norm.vmin = float(values.min()) if vmin is None else vmin
            shared_norm.vmax = float(values.max()) if vmax is None else vmax
        shared_norm.callbacks.process("changed")
    return fig
