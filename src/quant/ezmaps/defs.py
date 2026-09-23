from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from typing import TYPE_CHECKING

from .centroid import centroid
from .boundary import boundary_points, fit_boundary
from .pole import find_estimated_poles, find_true_poles, BoundaryPoint
from .midline import midline

if TYPE_CHECKING:
    from src.defs.fov import BoundingBox


class CellMap:
    mask: np.ndarray
    # What region of the FOV does the mask cover?
    bounding_box: BoundingBox

    centroid: np.ndarray

    estimated_poles: list[BoundaryPoint]
    true_poles: list[BoundaryPoint]

    boundary_by_theta: BSpline
    boundary_by_arc_length: BSpline
    perimeter: float

    midline_by_arc_length: BSpline
    midline_length: float

    def __init__(self, bounding_box, mask, blank = False):
        self.mask = mask
        self.bounding_box = bounding_box

        if blank:
            return

        self.centroid = centroid(mask)

        (self.boundary_by_theta, self.boundary_by_arc_length, self.perimeter) = (
            fit_boundary(self.centroid, boundary_points(self.mask))
        )

        self.estimated_poles = find_estimated_poles(
            self.boundary_by_arc_length, self.perimeter, self.centroid
        )

        self.midline_by_arc_length, self.midline_length = midline(
            self.estimated_poles,
            self.boundary_by_theta,
            self.boundary_by_arc_length,
            self.perimeter,
            self.centroid,
        )

        self.true_poles = find_true_poles(
            self.estimated_poles,
            self.boundary_by_arc_length,
            self.perimeter,
            self.midline_by_arc_length,
            self.midline_length,
            self.centroid,
        )

    def _cell_point_coordinates(self, cell_points):
        projection_als = np.asarray(
            [point.midline_position_px for point in cell_points], dtype=float
        )
        distances = np.asarray(
            [point.offset_from_midline_px for point in cell_points], dtype=float
        )

        projection_points = self.midline_by_arc_length(projection_als)
        tangents = self.midline_by_arc_length.derivative()(projection_als)
        tangent_lengths = np.linalg.norm(tangents, axis=1)
        normals = np.c_[-tangents[:, 1], tangents[:, 0]] / tangent_lengths[:, None]
        points = projection_points + distances[:, None] * normals

        return points, projection_points

    def _plot_cell_points(self, ax, cell_points, scale=1.0):
        cell_points = list(cell_points)

        if len(cell_points) == 0:
            return

        points, projection_points = self._cell_point_coordinates(cell_points)

        points = points * scale
        projection_points = projection_points * scale

        ax.plot(points[:, 1], points[:, 0], "bo")
        ax.plot(projection_points[:, 1], projection_points[:, 0], "mo")

        for point, projection_point in zip(points, projection_points):
            ax.plot(
                [point[1], projection_point[1]],
                [point[0], projection_point[0]],
                color="tab:gray",
                linewidth=1,
            )

    def show(
        self,
        show_mask=True,
        ax=None,
        extend_midline_by=1,
        render=True,
        cell_points=None,
        scale=1.0,
    ):
        if scale <= 0:
            raise ValueError("scale must be positive")

        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        if show_mask:
            height, width = self.mask.shape
            ax.imshow(
                self.mask,
                extent=(0, width * scale, height * scale, 0),
            )

        boundary_als = np.linspace(0, self.perimeter, num=300)
        boundary_points = self.boundary_by_arc_length(boundary_als)
        boundary_points = boundary_points * scale
        ax.plot(boundary_points[:, 1], boundary_points[:, 0], linewidth=1)

        estimated_pole_als = [pole.arc_length for pole in self.estimated_poles]
        estimated_pole_points = self.boundary_by_arc_length(estimated_pole_als)
        estimated_pole_points = estimated_pole_points * scale
        ax.plot(estimated_pole_points[:, 1], estimated_pole_points[:, 0], "rx")

        true_pole_als = [pole.arc_length for pole in self.true_poles]
        true_pole_points = self.boundary_by_arc_length(true_pole_als)
        true_pole_points = true_pole_points * scale
        ax.plot(true_pole_points[:, 1], true_pole_points[:, 0], "ro")

        midline_start = -self.midline_length / 2
        midline_end = self.midline_length / 2

        midline_als = np.linspace(midline_start, midline_end, num=200)
        midline_points = self.midline_by_arc_length(midline_als)
        midline_points = midline_points * scale
        ax.plot(midline_points[:, 1], midline_points[:, 0], color="tab:green")

        lower_midline_als = np.linspace(
            midline_start - extend_midline_by, midline_start, num=50
        )
        lower_midline_points = self.midline_by_arc_length(lower_midline_als)
        lower_midline_points = lower_midline_points * scale
        ax.plot(
            lower_midline_points[:, 1], lower_midline_points[:, 0], color="tab:orange"
        )

        upper_midline_als = np.linspace(
            midline_end, midline_end + extend_midline_by, num=50
        )
        upper_midline_points = self.midline_by_arc_length(upper_midline_als)
        upper_midline_points = upper_midline_points * scale
        ax.plot(
            upper_midline_points[:, 1], upper_midline_points[:, 0], color="tab:orange"
        )

        if cell_points is not None:
            self._plot_cell_points(ax, cell_points, scale=scale)

        # Preserve the physical aspect ratio of the map regardless of the
        # figure's dimensions.
        ax.set_aspect("equal")

        if render:
            plt.show()

        return fig, ax
