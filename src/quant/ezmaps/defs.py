import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from .centroid import centroid
from .boundary import boundary_points, fit_boundary
from .pole import find_estimated_poles, find_true_poles, BoundaryPoint
from .midline import midline
from src.defs.fov import Region

class CellMap:

    mask: np.ndarray
    # What region of the FOV does the mask cover?
    region: Region 

    centroid: np.ndarray

    estimated_poles: list[BoundaryPoint]
    true_poles: list[BoundaryPoint]
    
    boundary_by_theta: BSpline
    boundary_by_arc_length: BSpline
    perimeter: float
    
    midline_by_arc_length: BSpline
    midline_length: float

    def __init__(self, region, mask):
        self.mask = mask
        self.region = region

        self.centroid = centroid(mask)

        (
            self.boundary_by_theta,
            self.boundary_by_arc_length,
            self.perimeter
        ) = fit_boundary(
            self.centroid,
            boundary_points(self.mask)
        )

        self.estimated_poles = find_estimated_poles(
            self.boundary_by_arc_length,
            self.perimeter,
            self.centroid
        )

        self.midline_by_arc_length, self.midline_length = midline(
            self.estimated_poles,
            self.boundary_by_theta,
            self.boundary_by_arc_length,
            self.perimeter,
            self.centroid
        )

        self.true_poles = find_true_poles(
            self.estimated_poles,
            self.boundary_by_arc_length,
            self.perimeter,
            self.midline_by_arc_length,
            self.midline_length,
            self.centroid
        )

    def _cell_point_coordinates(self, cell_points):
        projection_als = np.asarray(
            [point.midline_position_px for point in cell_points],
            dtype = float
        )
        distances = np.asarray(
            [point.offset_from_midline_px for point in cell_points],
            dtype = float
        )

        projection_points = self.midline_by_arc_length(projection_als)
        tangents = self.midline_by_arc_length.derivative()(projection_als)
        tangent_lengths = np.linalg.norm(tangents, axis = 1)
        normals = np.c_[-tangents[:, 1], tangents[:, 0]] / tangent_lengths[:, None]
        points = projection_points + distances[:, None] * normals

        return points, projection_points

    def _plot_cell_points(self, ax, cell_points):
        cell_points = list(cell_points)

        if len(cell_points) == 0:
            return

        points, projection_points = self._cell_point_coordinates(cell_points)

        point_rows, point_cols = points.T
        projection_rows, projection_cols = projection_points.T

        ax.plot(point_cols, point_rows, "bo")
        ax.plot(projection_cols, projection_rows, "mo")

        for point, projection_point in zip(points, projection_points):
            ax.plot(
                [point[1], projection_point[1]],
                [point[0], projection_point[0]],
                color = "tab:gray",
                linewidth = 1
            )

    def show(self, ax = None, extend_midline_by = 1, render = True, cell_points = None):

        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        ax.imshow(self.mask)

        boundary_als = np.linspace(0, self.perimeter, num = 300)
        boundary_row, boundary_col = self.boundary_by_arc_length(boundary_als).T
        ax.plot(boundary_col, boundary_row, linewidth = 1)

        estimated_pole_als = [pole.arc_length for pole in self.estimated_poles]
        estimated_pole_row, estimated_pole_col = self.boundary_by_arc_length(estimated_pole_als).T
        ax.plot(estimated_pole_col, estimated_pole_row, "rx")

        true_pole_als = [pole.arc_length for pole in self.true_poles]
        true_pole_row, true_pole_col = self.boundary_by_arc_length(true_pole_als).T
        ax.plot(true_pole_col, true_pole_row, "ro")

        midline_start = -self.midline_length / 2
        midline_end = self.midline_length / 2

        midline_als = np.linspace(midline_start, midline_end, num = 200)
        midline_row, midline_col = self.midline_by_arc_length(midline_als).T
        ax.plot(midline_col, midline_row, color = "tab:green")

        lower_midline_als = np.linspace(midline_start - extend_midline_by, midline_start, num = 50)
        lower_midline_row, lower_midline_col = self.midline_by_arc_length(lower_midline_als).T
        ax.plot(lower_midline_col, lower_midline_row, color = "tab:orange")

        upper_midline_als = np.linspace(midline_end, midline_end + extend_midline_by, num = 50)
        upper_midline_row, upper_midline_col = self.midline_by_arc_length(upper_midline_als).T
        ax.plot(upper_midline_col, upper_midline_row, color = "tab:orange")

        if cell_points is not None:
            self._plot_cell_points(ax, cell_points)

        if render:
            plt.show()

        return fig, ax
