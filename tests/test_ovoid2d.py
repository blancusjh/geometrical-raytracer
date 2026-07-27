"""Cartesian oval (2-D): aperture clipping and exact-curve drawing."""

import numpy as np

from raytracer import CartesianOvalSurface
from raytracer.optics.ray import Ray
from raytracer.surfaces.cartesian_oval import cartesian_oval_implicit

Z0, ZI, N0, NI = -30.0, 10.0, 1.0, 1.7


def test_semidiameter_clips_rays_beyond_the_aperture():
    oval = CartesianOvalSurface(z0=Z0, zi=ZI, n_exterior=N0, n_interior=NI, semidiameter=2.0)

    within = Ray(origin=np.array([Z0, 1.0]), direction=np.array([1.0, 0.0]))
    assert oval.hit(within) is not None

    beyond = Ray(origin=np.array([Z0, 3.0]), direction=np.array([1.0, 0.0]))
    assert oval.hit(beyond) is None


def test_drawn_curve_matches_the_traced_fermat_surface():
    """Regression for the sigma_parametric() placeholder: as_points() must lie
    on the same surface cartesian_oval_implicit()/hit() actually trace."""

    oval = CartesianOvalSurface(z0=Z0, zi=ZI, n_exterior=N0, n_interior=NI)
    points = oval.as_points(samples=64)
    assert points.shape[0] > 10  # not the degenerate near-empty placeholder curve

    residuals = [
        cartesian_oval_implicit(z, abs(r), Z0, ZI, N0, NI) for z, r in points
    ]
    assert np.max(np.abs(residuals)) < 1e-8
