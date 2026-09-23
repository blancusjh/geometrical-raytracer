"""An absorbing meridional aperture stop, extending outside its clear opening."""

import numpy as np

from ..math.transforms import RigidTransform
from .surface import Surface


class ApertureStopSurface(Surface):
    def __init__(self, vertex, radius, *, inner_radius=0.0, surface_id="stop"):
        super().__init__(surface_id=surface_id, absorbing=True)
        self.frame = RigidTransform.from_angle_2d(np.asarray(vertex), 0.0)
        self.radius = float(radius)
        self.inner_radius = float(inner_radius)
        self.quadratic_form = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    def implicit(self, point):
        return float(point[0])

    def implicit_gradient(self, point):
        return np.array([1.0, 0.0])

    def within_aperture(self, point):
        height = abs(point[1])
        return height > self.radius or height < self.inner_radius

    def polyline(self, samples=4):
        r = self.radius
        points = np.array([[0, -2 * r], [0, -r], [np.nan, np.nan], [0, r], [0, 2 * r]])
        return self.frame.to_world(points)
