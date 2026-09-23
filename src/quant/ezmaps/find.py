import numpy as np
from .defs import CellMap


class CellPoint:
    midline_position_px: float
    offset_from_midline_px: float
    vector_from_centroid_px: np.ndarray

    def __init__(self, midline_position_px: float, offset_from_midline_px: float, vector_from_centroid_px: np.ndarray):
        self.midline_position_px = midline_position_px
        self.offset_from_midline_px = offset_from_midline_px
        self.vector_from_centroid_px = vector_from_centroid_px


# Given a CellMap, find where it would land on the midline and
# how far away it is - this roughly matches the Spideymaps l and r coordinates
def locate(cell_map: CellMap, point: np.ndarray, rounds=5) -> CellPoint:
    """Returns a CellPoint representing the location.

    Arguments:
    cell_map -- the map for this particular cell.
    point -- [row, col] position to be located using the cell map.
    rounds -- how many iterations of approximation should be used.
    """
    pole_als = [pole.arc_length for pole in cell_map.true_poles]
    pole_a, pole_b = cell_map.boundary_by_arc_length(pole_als)
    pole_vec = pole_b - pole_a
    pole_unit = pole_vec / np.linalg.norm(pole_vec)

    # First: project the point onto the vector connecting the two cell poles
    # This is just an approximation that assumes the midline is completely
    # linear, which it is not, but it's useful
    pole_projection = (
        pole_a
        + (np.dot(point - pole_a, pole_vec) / np.dot(pole_vec, pole_vec)) * pole_vec
    )

    al = np.dot(pole_projection - cell_map.centroid, pole_unit)

    midline = cell_map.midline_by_arc_length
    midline_prime = midline.derivative()

    try:
        midline_double_prime = midline.derivative(2)
    except ValueError:
        midline_double_prime = None

    # This is basically Newton's method
    # Increase the number of rounds to get a more accurate location
    for _ in range(rounds):
        midline_point = midline(al)
        tangent = midline_prime(al)
        # Is the tangent perpendicular to the vector connecting the target
        # point to the midline? if so, good, and this will be zero
        # if not, this will capture which direction we should move
        # to minimize the length of midline_point - point; if that is
        # in the same direction as tangent, al is too great
        # otherwise, it is too little
        normal_error = np.dot(midline_point - point, tangent)

        if midline_double_prime is None:
            normal_slope = np.dot(tangent, tangent)
        else:
            curvature = midline_double_prime(al)
            normal_slope = np.dot(tangent, tangent) + np.dot(
                midline_point - point, curvature
            )

        if normal_slope == 0:
            break

        al -= normal_error / normal_slope

    midline_point = midline(al)
    tangent = midline_prime(al)
    tangent = tangent / np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    offset_from_midline = np.dot(point - midline_point, normal)

    return CellPoint(
        float(al),
        float(offset_from_midline),
        np.asarray(point, dtype=float) - cell_map.centroid,
    )
