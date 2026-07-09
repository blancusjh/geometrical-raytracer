"""Cartesian ovoid (2-D): aperture clipping and exact-curve drawing."""

import numpy as np

from raytracer import CartesianOvoid2D
from raytracer.geometry.ovoid2d import fermat_ovoid_F
from raytracer.nonseq.rays import Ray2D

Z0, ZI, N0, NI = -30.0, 10.0, 1.0, 1.7


def test_semidiameter_clips_rays_beyond_the_aperture():
    ovoid = CartesianOvoid2D(z0=Z0, zi=ZI, n_exterior=N0, n_interior=NI, semidiameter=2.0)

    within = Ray2D(origin=np.array([Z0, 1.0]), direction=np.array([1.0, 0.0]))
    assert ovoid.intersect(within) is not None

    beyond = Ray2D(origin=np.array([Z0, 3.0]), direction=np.array([1.0, 0.0]))
    assert ovoid.intersect(beyond) is None


def test_drawn_curve_matches_the_traced_fermat_surface():
    """Regression for the sigma_parametric() placeholder: as_points() must lie
    on the same surface fermat_ovoid_F()/intersect() actually trace."""

    ovoid = CartesianOvoid2D(z0=Z0, zi=ZI, n_exterior=N0, n_interior=NI)
    points = ovoid.as_points(samples=64)
    assert points.shape[0] > 10  # not the degenerate near-empty placeholder curve

    residuals = [
        fermat_ovoid_F(z, abs(r), Z0, ZI, N0, NI) for z, r in points
    ]
    assert np.max(np.abs(residuals)) < 1e-8
