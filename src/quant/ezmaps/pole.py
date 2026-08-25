from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize

# Represents a point on the boundary
# of a cell using two interconvertible
# coordinates for convenience
@dataclass
class BoundaryPoint:
    arc_length: float
    theta: float

# Estimate the pole locations using a
# heuristic about how far apart they should be
def find_estimated_poles(
    boundary_by_arc_length: BSpline,
    perimeter: float,
    centroid: np.ndarray,
    n = 200
) -> list[BoundaryPoint]:
    """Returns a list of 2 [row, col] points representing the estimated pole locations of the cell.

    Arguments:
    boundary_by_arc_length -- BSpline that produces a [row, col] point along the cell boundary given the arc length of the segment of 
    the boundary from theta = 0 to this point.
    perimeter -- the arc length of the boundary of the cell.
    centroid -- the [row, col] position of the center of the cell
    n -- how many points should be sampled along half of the perimeter of the cell (more may improve the accuracy).
    
    """
    
    # Given one point on the cell boundary specified by
    # an arc length, return the opposite point,
    # i.e. the point halfway around the cell as you're walking
    # around the perimeter. Note that this need not be
    # at the opposite angle. For an exaggerated example,
    # if your cell looks like this
    #
    #   -------
    #  /       \
    # /   /\    \
    # -*--  --*--
    #
    # the poles marked with stars might be opposite but they're on the
    # same side of the cell centroid
    def opposite(al):
        return (al + (perimeter / 2)) % perimeter
    
    # Generate points around half of the perimeter and find their opposites
    als = np.linspace(0, perimeter / 2, num = n)
    ops = [opposite(x) for x in als]
    coordinates_1 = boundary_by_arc_length(als)
    coordinates_2 = boundary_by_arc_length(ops)

    # Calculate the distances between the two potential poles
    dist = np.sqrt(
        np.sum(np.square(coordinates_1 - coordinates_2), axis=1)
    )

    # The poles will theoretically have the greatest straight-line distance
    # between them of the any opposite points
    max_idx = np.argmax(dist)

    # We want the lowest arc length pole first for convenience
    pole_als = sorted([als[max_idx], ops[max_idx]])
    pole_coordinates = boundary_by_arc_length(pole_als)

    pole_thetas = np.atan2(pole_coordinates[:, 1] - centroid[1], pole_coordinates[:, 0] - centroid[0])

    return [
        BoundaryPoint(
            arc_length = float(pole_als[idx]),
            theta = float(pole_thetas[idx])
        )
        for idx in range(len(pole_als))
    ]

# This is a more sophisticated pole-finding method that refines the previous one
# I discovered that this works better because sometimes the midline
# did not pass through both estimated poles, but where it did intersect the outline was
# closer to the true pole than the estimate
def find_true_poles(
    estimated_poles: list[BoundaryPoint],
    boundary_by_arc_length: BSpline,
    perimeter: float,
    midline_by_arc_length: BSpline,
    midline_length: float,
    centroid: np.ndarray,
    search_distance = 10
) -> list[BoundaryPoint]:

    # Along the midline, you must travel from the cell center
    # forward half of the cell length or backward half of the cell
    # length to get to the poles
    midline_pole_als = [
        -midline_length / 2,
        midline_length / 2
    ]

    def true_pole(estimated_pole, midline_pole_al):
        # Use the squared distance for convenience
        # Computes the distance between on the cell boundary
        # according to its arc length coordinate and a point on the midline
        def dist(args):
            boundary_al, midline_al = args
            diff = (
                boundary_by_arc_length(boundary_al % perimeter)
                - midline_by_arc_length(midline_al)
            )
            return np.dot(diff, diff)

        # The "true pole" is just where the midline crosses the
        # cell boundary - this is found by minimization of the distance above
        result = minimize(
            dist,
            [estimated_pole.arc_length, midline_pole_al],
            bounds = [
                (0, perimeter),
                (
                    midline_pole_al - search_distance,
                    midline_pole_al + search_distance
                )
            ],
            method = "L-BFGS-B" # Kinda efficient
        )

        boundary_al, midline_al = result.x
        boundary_al = boundary_al % perimeter
        point = boundary_by_arc_length(boundary_al)
        theta = np.atan2(point[1] - centroid[1], point[0] - centroid[0])

        return BoundaryPoint(
            arc_length = float(boundary_al),
            theta = float(theta)
        )

    return [
        true_pole(estimated_pole, midline_pole_al)
        for estimated_pole, midline_pole_al in zip(estimated_poles, midline_pole_als)
    ]
