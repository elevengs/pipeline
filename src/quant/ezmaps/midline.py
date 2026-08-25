import numpy as np
from scipy.interpolate import BSpline, make_interp_spline
from .pole import BoundaryPoint

# The basic idea is to slice the cell length-wise in half into top and bottom
# Like this:
#
#        -#--------------
#       / | -            \
#      /  |   -           \
#     *---.-----c----------*
#      \  |    -          /
#       \ | -            /
#        -#--------------
#
# If you look a specific angle away from the pole in both directions,
# see where on the cell bondary those angles lands you, and average, that result will be
# close to the cell midline.
# 
# Based on this, this function returns a spline which takes in
# a signed length and returns a point on the cell midline
# that distance from the cell center.
#
# This works extremely well in practice.
def midline(
    poles: list[BoundaryPoint],
    boundary_by_theta: BSpline,
    boundary_by_arc_length: BSpline,
    perimeter: float,
    centroid: np.ndarray,
    n = 200,
    extension_length = 10,
    n_extension = 50
) -> tuple[BSpline, float]:
    """Returns (midline_spline, midline_length).
    
    midline_spline is a BSpline that returns the [row, col] position of a point on the midline of the cell
    given the signed distance traveled from the cell center to that point (backwards being negative, forwards being positive).
    This spline is also extended linearly past both of the cell poles.
    midline_length is the length of the midline strictly between the two cell poles.

    Arguments:
    poles -- list of 2 points along the boundary of the cell that represent the cell poles.
    boundary_by_theta -- BSpline that produces a [row, col] point along the cell boundary given an angle theta from the centroid, 
    where theta = 0 points in the positive row direction.
    boundary_by_arc_length -- BSpline that produces a [row, col] point along the cell boundary given the arc length of the segment of 
    the boundary from theta = 0 to this point.
    perimeter -- the arc length of the boundary of the cell.
    n -- how many points should be sampled along half of the boundary of the cell (more may be more accurate).
    extension_length -- how far past the poles the points used for extending the interpolation spline should be.
    n_extension -- how many points along each extended line should be used for creating the interpolation spline. 
    """
    pole_a = poles[0]
    # Get angles for points on the upper and lower halves of the cell boundary
    # (Takes n points on each half)
    top_thetas = np.linspace(*[pole.theta for pole in poles], num = n)
    bottom_thetas = pole_a.theta - (top_thetas - pole_a.theta)

    top_coordinates = boundary_by_theta(top_thetas)
    bottom_coordinates = boundary_by_theta(bottom_thetas)

    # Take the midpoint of the line segment connecting the
    # points on each half, as shown in my beautiful diagram above
    midline_coordinates = (top_coordinates + bottom_coordinates) / 2

    # Find the distances between the points
    # after cumulative summation, this represents the arc length
    # along the midline from pole a
    midline_al = [0,
        *np.cumsum(
            np.sqrt(
                np.sum(np.square(np.diff(midline_coordinates, axis=0)), axis=1)
            )
        )
    ]

    # How long is the cell midline?
    midline_length = midline_al[len(midline_al) - 1]

    # Again, this is an interpolation spline because
    # the points we do have are considered exact
    mid_spline = make_interp_spline(
        midline_al,
        midline_coordinates
    )

    # In order to locate foci that are past the poles of the cell,
    # we extend the midlines using the tangent line at the poles
    start_point = mid_spline(0)
    end_point = mid_spline(midline_length)

    start_derivative = mid_spline.derivative()(0)
    end_derivative = mid_spline.derivative()(midline_length)
    
    start_derivative /= np.linalg.norm(start_derivative)
    end_derivative /= np.linalg.norm(end_derivative)

    start_radial = start_point - centroid
    end_radial = end_point - centroid

    # If the midline is going the same direction as the radial at the start point,
    # reverse the direction because it needs to go the opposite way
    # (e.g. as you increase the arc length value, the midline point should move
    # TOWARDS the cell centroid, not away from it)
    if np.dot(start_derivative, start_radial) > 0:
        start_derivative *= -1

    # Reverse of above logic
    if np.dot(end_derivative, end_radial) < 0:
        end_derivative *= -1

    # Set up for yet another interpolation spline using these extended lines
    lower_als = np.linspace(-extension_length, 0, num = n_extension, endpoint = False)
    upper_als = np.linspace(midline_length, midline_length + extension_length, num = n_extension)[1:]

    lower_points = start_point + lower_als[:, None] * start_derivative
    upper_points = end_point + (upper_als[:, None] - midline_length) * end_derivative

    extended_al = [
        *lower_als,
        *midline_al,
        *upper_als
    ]

    # Cell center should by roughly zero on the midline
    extended_al = [
        x - (midline_length / 2)
            for x in extended_al
    ]

    extended_points = np.r_[
        lower_points,
        midline_coordinates,
        upper_points
    ]

    # Interpolate again - this should be very close to the original spline
    # at every point, and it should extend infinitely past the cell poles
    # a k = 1 spline will stay linear there whereas higher degrees will do crazy stuff
    return make_interp_spline(extended_al, extended_points, k = 1), midline_length    
