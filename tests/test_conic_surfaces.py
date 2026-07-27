"""Geometric ground truths for the 2-D conic surfaces."""

import numpy as np
import pytest

from raytracer import (
    BranchingTracer,
    CircleSurface,
    EllipseSurface,
    ParabolaSurface,
    ParallelSource,
    PointSource,
    Ray,
    TraceConfig,
)


def test_circle_intersection_distance():
    circle = CircleSurface(radius=2.0)
    ray = Ray(origin=[-5.0, 0.0], direction=[1.0, 0.0])
    hit = circle.hit(ray)
    assert hit is not None
    assert hit.distance == pytest.approx(3.0, abs=1e-12)
    assert hit.point == pytest.approx(np.array([-2.0, 0.0]), abs=1e-12)
    assert abs(hit.normal @ np.array([1.0, 0.0])) == pytest.approx(1.0, abs=1e-12)


def test_ellipse_focus_to_focus_reflection():
    """Rays from one ellipse focus reflect through the other focus."""

    a, b = 4.0, 2.5
    c = np.sqrt(a**2 - b**2)
    # ConicSurface places the *focus* argument at the polar-form origin.
    ellipse = EllipseSurface(semi_major=a, semi_minor=b, focus=[0.0, 0.0])
    other_focus = np.array([-2.0 * c, 0.0])

    source = PointSource(
        origin=np.array([0.0, 0.0]),
        axis_direction=np.array([1.0, 0.0]),
        aperture=np.deg2rad(300.0),
        samples=41,
    )
    tree = BranchingTracer([ellipse], TraceConfig(max_generations=2)).trace([source])

    checked = 0
    for node in tree.nodes():
        if node.generation != 2 or node.intersection is None:
            continue
        # Reflected ray should pass through the other focus: the segment
        # origin->hit must be collinear with origin->other_focus.
        to_hit = node.intersection.point - node.ray.origin
        to_focus = other_focus - node.ray.origin
        cross = to_hit[0] * to_focus[1] - to_hit[1] * to_focus[0]
        assert abs(cross) < 1e-7 * np.linalg.norm(to_hit) * np.linalg.norm(to_focus)
        checked += 1
    assert checked > 10


def test_parabola_focuses_collimated_beam():
    """A collimated beam along the axis reflects through the parabola focus."""

    p = 2.0  # semi-latus rectum; focus at origin in polar form
    parabola = ParabolaSurface(p=p)
    source = ParallelSource(
        origin=np.array([-6.0, 0.0]),
        direction=np.array([1.0, 0.0]),
        width=2.5,
        samples=21,
    )
    tree = BranchingTracer([parabola], TraceConfig(max_generations=2)).trace([source])

    checked = 0
    for node in tree.nodes():
        if node.generation != 2:
            continue
        # Reflected ray must pass through the focus (origin).
        origin, direction = node.ray.origin, node.ray.direction
        # Perpendicular distance from origin-point to the ray line:
        perp = abs(-direction[1] * origin[0] + direction[0] * origin[1])
        assert perp < 1e-9
        checked += 1
    assert checked > 10


def test_semidiameter_clips_rays_beyond_the_aperture():
    """A ray that would hit the conic outside its clear aperture must miss."""

    circle = CircleSurface(radius=5.0, center=np.array([10.0, 0.0]), semidiameter=2.0)

    within = Ray(origin=[0.0, 1.0], direction=[1.0, 0.0])
    assert circle.hit(within) is not None

    beyond = Ray(origin=[0.0, 3.0], direction=[1.0, 0.0])
    assert circle.hit(beyond) is None


def test_unset_semidiameter_does_not_clip():
    """Default (None) semidiameter keeps the old, unclipped behaviour."""

    circle = CircleSurface(radius=5.0, center=np.array([10.0, 0.0]))
    ray = Ray(origin=[0.0, 4.9], direction=[1.0, 0.0])
    assert circle.hit(ray) is not None
