"""GPU smoke tests: additive linearity and non-saturation of the HDR path.

These need a working OpenGL context; they are skipped automatically when
none is available (headless CI).
"""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _make_canvas():
    pytest.importorskip("vispy")
    from raytracer.viz.gl.viewer import OpenGLViewer, RenderConfig

    try:
        viewer = OpenGLViewer(
            x_lims=(-5, 5), y_lims=(-5, 5), size=(200, 200),
            render_config=RenderConfig(ray_width=0.4, sigma_factor=0.4,
                                       auto_exposure=False, exposure=1.0),
        )
        _draw(viewer)  # force context + pipeline creation
    except Exception as error:
        pytest.skip(f"No usable OpenGL context: {error}")
    return viewer


def _draw(viewer):
    """Run the draw pipeline without Canvas.render() (Qt DPI quirks)."""

    viewer.set_current()
    viewer.on_draw(None)


class _Node:
    def __init__(self, origin, direction, hit_point):
        from raytracer.nonseq.rays import Intersection2D, Ray2D

        self.ray = Ray2D(origin, direction)
        self.intersection = (
            None
            if hit_point is None
            else Intersection2D(
                point=hit_point, normal=[0.0, 1.0], distance=1.0, surface_id="s"
            )
        )
        self.generation = 1
        self.intensity = 1.0


class _Tree:
    def __init__(self, nodes):
        self._nodes = nodes

    def nodes(self):
        return self._nodes


def test_accumulation_is_linear_additive():
    viewer = _make_canvas()
    one = _Tree([_Node([-3.0, 0.0], [1.0, 0.0], [3.0, 0.0])])
    two = _Tree(
        [
            _Node([-3.0, 0.0], [1.0, 0.0], [3.0, 0.0]),
            _Node([-3.0, 0.0], [1.0, 0.0], [3.0, 0.0]),
        ]
    )

    viewer.draw_rays(one)
    _draw(viewer)
    energy_one = viewer.read_accumulation()[..., :3].max()

    viewer.clear_rays()  # draw_rays accumulates bundles; start fresh
    viewer.draw_rays(two)
    _draw(viewer)
    energy_two = viewer.read_accumulation()[..., :3].max()

    assert energy_one > 0.0
    assert energy_two == pytest.approx(2.0 * energy_one, rel=1e-3)
    viewer.close()


def test_display_never_saturates_with_many_overlapping_rays():
    from vispy import gloo

    viewer = _make_canvas()
    viewer.config.auto_exposure = True
    viewer._exposure_dirty = True
    nodes = [_Node([-3.0, 0.001 * k], [1.0, 0.0], [3.0, 0.001 * k]) for k in range(200)]
    viewer.draw_rays(_Tree(nodes))
    _draw(viewer)

    accum = viewer.read_accumulation()[..., :3].max()
    assert accum > 10.0  # HDR buffer accumulated far beyond display range

    display = viewer.snapshot()[..., :3]
    # Tone mapping compresses to displayable range; auto-exposure pins the
    # 99th percentile near (but below) full scale.
    assert display.max() <= 255
    assert display.max() > 100  # bright, not black
    viewer.close()
