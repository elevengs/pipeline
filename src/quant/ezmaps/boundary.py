# Here's the plan
#
# 1. Take in the cell mask as an ndarray
# 2. Use scikit-image contour to get an array of points on the outside
# 3. Fit that array of points with a bspline to get the cell boundary spline
# 4. Find the two opposite points on the spline with the longest distance between them
#    This will split the boundary into a "top bun" and "lower bun"
# 5. Draw the midline as the average of the two halves of the spline

import numpy as np
from skimage.measure import find_contours
from scipy.interpolate import BSpline, make_interp_spline, make_smoothing_spline

def boundary_points(mask: np.ndarray) -> np.ndarray:
    """Returns ``np.ndarray`` of shape ``(n, 2)`` where ``n`` is the number of points around the mask edge.

    The index ``0`` along dimension ``1`` contains the row indices, whereas index ``1`` contains
    the column indices.
    """
    return find_contours(mask, 0.5)[0]

def fit_boundary(
    centroid: np.ndarray,
    boundary_pts: np.ndarray,
    n_theta_samples: int = 1000 
) -> tuple[BSpline, BSpline, float]:
    """Returns ``(boundary_by_theta, boundary_by_arc_length, perimeter)``.

    ``boundary_by_theta`` is a ``BSpline`` which returns ``(row, col)`` coordinates along the boundary of the cell
    given the angle of the boundary point from the centroid, where ``theta = 0`` points in the positive row direction. 

    ``boundary_by_arc_length`` parametrizes the same boundary in terms of the arc length, starting from the point on 
    the boundary where ``theta = 0``.

    ``perimeter`` is the arc length of the boundary of the cell.

    Positional arguments:
    ``centroid`` -- ``[row, col]`` position of the cell center
    ``boundary_pts`` -- ``[[row, col], ...]`` positions of the boundary points
    ``n_theta_samples`` -- number of ``theta`` values to use when sampling from ``boundary_by_theta`` to reparametrize by arc length
    """

    # Closed contours repeat the first point at the end
    # drop only that duplicate so theta and point coordinates stay paired.
    if np.allclose(boundary_pts[0], boundary_pts[-1]):
        boundary_pts = boundary_pts[:-1]

    # Angle from the centroid as to the positive row direction
    # the absolute start angle doesn't matter
    theta = [
        np.atan2(pt[1] - centroid[1], pt[0] - centroid[0])
            for pt in boundary_pts
    ]

    sorted_theta, sorted_row, sorted_col = list(
        # zip(*_) is the inverse of zip(_)
        # so this is zip -> sort by angle -> unzip
        zip(
            *sorted(
                zip(
                    theta,
                    boundary_pts[:, 0],

                    boundary_pts[:, 1]
                ),
                key = lambda x: x[0]
            )
        )
    )

    # Remove non-ascending theta values
    # SciPy gets mad when you try to fit
    # a spline with non-ascending x values
    asc_theta, asc_row, asc_col = ([], [], [])
    for i in range(len(sorted_theta)):         
        if len(asc_theta) == 0 or sorted_theta[i] > asc_theta[len(asc_theta) - 1]:
            asc_theta.append(sorted_theta[i])
            asc_row.append(sorted_row[i])
            asc_col.append(sorted_col[i])

    # Repeat the sequence on either side so it is periodic
    # This helps the spline fit at the endpoints (-pi, +pi) more accurately
    asc_theta = [
        *[theta - 2 * np.pi for theta in asc_theta],
        *asc_theta,
        *[theta + 2 * np.pi for theta in asc_theta],
    ]
    asc_row = [*asc_row, *asc_row, *asc_row]
    asc_col = [*asc_col, *asc_col, *asc_col]

    # Fit the outline of the cell by theta
    # This will allow you to query an angle (say, 90 degrees)
    # and find what point on the outline of the cell falls at that angle
    boundary_by_theta = make_smoothing_spline(
        asc_theta,
        np.c_[asc_row, asc_col],
        # this value is smoothing
        # increase to get a smoother spline at the potential
        # expense of accuracy
        lam = 0.01
    )

    # Next: we have the outline parametrized by theta,
    # but we want it parametrized by arc length.
    
    # There must be an odd number of points so that there is a middle value
    use_n_theta_samples = n_theta_samples if n_theta_samples % 2 == 1 else n_theta_samples + 1
    theta_new = np.linspace(- 3 * np.pi, 3 * np.pi, num = use_n_theta_samples)
    row_new, col_new = boundary_by_theta(theta_new).T

    # Prepend 0 so that this has the same length as phi_new
    # otherwise, with np.diff it would be 1 shorter.
    # This makes logical sense because the first point
    # has a cumulative arc length of zero, not some positive number.
    arc_length = [0, *np.cumsum(
        np.sqrt(
            np.square(np.diff(row_new)) +
            np.square(np.diff(col_new))
        )
    )]

    # Translate arc length so that 0 lies at theta = 0
    arc_length_at_0 = arc_length[int((len(arc_length) - 1) / 2)]
    perimeter = np.max(arc_length) / 3
    arc_length = [x - arc_length_at_0 for x in arc_length]
    
    # Interpolate instead of smoothing
    # This is necessary when the values you have
    # are exactly accurate and you need to make
    # inferences about the values in between
    boundary_by_arc_length = make_interp_spline(
        arc_length,
        np.c_[row_new, col_new]
    )

    return boundary_by_theta, boundary_by_arc_length, perimeter
