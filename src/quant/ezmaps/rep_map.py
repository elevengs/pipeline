from collections.abc import Iterable

import numpy as np

from src.defs import Cell
from src.quant.ezmaps.boundary import fit_boundary
from src.quant.ezmaps.midline import midline
from src.quant.ezmaps.pole import find_estimated_poles, find_true_poles
from .defs import CellMap


def maps_to_rep_map(
    cells: Iterable[Cell], n_theta=500
) -> tuple[CellMap, dict[str, np.ndarray]]:
    """Average cell maps after aligning their true poles.

    Returns ``(cell_map, transforms)`` where ``cell_map`` is the representative
    cell map and ``transforms`` maps each cell ID to its linear transform.

    The returned transforms operate on vectors in micrometers. To transform a
    focus vector, use:

        transforms[cell_id] @ focus.vector_from_centroid_um
    """
    cells = list(cells)
    cell_maps = [
        (
            cell.id,
            cell.map,
            cell.fov.image_dimensions.microns_per_pixel,
        )
        for cell in cells
    ]
    if not cell_maps:
        raise ValueError("at least one cell map is required")
    if n_theta < 3:
        raise ValueError("n_theta must be at least 3")

    rep_midline_length = np.median(
        [cell_map.midline_length * mpp for _, cell_map, mpp in cell_maps]
    )
    representative_mpp = np.median([mpp for _, _, mpp in cell_maps])

    rep_thetas = np.linspace(-np.pi, np.pi, num=n_theta, endpoint=False)
    rep_boundary_pts = []

    linear_transformations: dict[str, np.ndarray] = {}
    valid_cell_maps = []
    for cell_id, cell_map, mpp in cell_maps:
        pole_0_coords = (
            cell_map.boundary_by_arc_length(cell_map.true_poles[0].arc_length)
            - cell_map.centroid
        ) * mpp
        pole_1_coords = (
            cell_map.boundary_by_arc_length(cell_map.true_poles[1].arc_length)
            - cell_map.centroid
        ) * mpp
        pole_axis = pole_0_coords - pole_1_coords
        pole_distance = np.linalg.norm(pole_axis)
        if (
            not np.isfinite(pole_distance)
            or pole_distance <= np.finfo(float).eps
            or not np.isfinite(cell_map.midline_length)
            or cell_map.midline_length <= np.finfo(float).eps
        ):
            continue

        pole_axis /= pole_distance
        # Coordinates are (row, column), while plots use column as x and row
        # as y.  Map the pole axis to +column so representative cells plot
        # horizontally.  This matrix is a rotation; ``scale`` is one scalar,
        # so the scale remains uniform in both directions.
        rotation = np.asarray(
            [[pole_axis[1], -pole_axis[0]], [pole_axis[0], pole_axis[1]]]
        )
        cell_midline_length = cell_map.midline_length * mpp
        scale = rep_midline_length / cell_midline_length
        linear_transformations[cell_id] = scale * rotation
        valid_cell_maps.append((cell_id, cell_map, mpp))

    if not valid_cell_maps:
        raise ValueError("no cell maps have usable pole coordinates")

    for rep_theta in rep_thetas:
        total = np.zeros(2, dtype=float)

        # Take the arithmetic mean of the position relative to the centroid
        n_valid = 0
        for cell_id, cell_map, mpp in valid_cell_maps:
            pole_0 = cell_map.true_poles[0].theta
            src = cell_map.boundary_by_theta(pole_0 + rep_theta)
            M = linear_transformations[cell_id]
            dest = M @ ((src - cell_map.centroid) * mpp)
            total += dest
            n_valid += 1

        total /= n_valid

        rep_boundary_pts.append(total)

    # Using the rep boundary points, generate the rest of
    # the cell map according to the normal procedure
    # (except the mask and bounding box)

    rep_centroid = np.zeros(2, dtype=float)
    rep_boundary_by_theta, rep_boundary_by_arc_length, rep_perimeter = fit_boundary(
        rep_centroid, np.asarray(rep_boundary_pts)
    )

    rep_estimated_poles = find_estimated_poles(
        rep_boundary_by_arc_length, rep_perimeter, rep_centroid
    )

    rep_midline_by_arc_length, rep_midline_length = midline(
        rep_estimated_poles,
        rep_boundary_by_theta,
        rep_boundary_by_arc_length,
        rep_perimeter,
        rep_centroid,
        extension_length=int(round(10 * representative_mpp)),
    )

    rep_true_poles = find_true_poles(
        rep_estimated_poles,
        rep_boundary_by_arc_length,
        rep_perimeter,
        rep_midline_by_arc_length,
        rep_midline_length,
        rep_centroid,
        search_distance=int(round(10 * representative_mpp)),
    )

    # Empty mask

    rep_mask = np.empty((0, 0), dtype=bool)

    rep_bounding_box = np.full((2, 2), np.nan)

    rep_cell_map = CellMap(rep_bounding_box, rep_mask, blank=True)

    rep_cell_map.centroid = rep_centroid
    rep_cell_map.estimated_poles = rep_estimated_poles
    rep_cell_map.true_poles = rep_true_poles
    rep_cell_map.boundary_by_theta = rep_boundary_by_theta
    rep_cell_map.boundary_by_arc_length = rep_boundary_by_arc_length
    rep_cell_map.perimeter = rep_perimeter
    rep_cell_map.midline_by_arc_length = rep_midline_by_arc_length
    rep_cell_map.midline_length = rep_midline_length

    return rep_cell_map, linear_transformations
