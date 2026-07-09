"""Regression: multiple ray bundles drawn on one viewer must all render.

draw_rays/draw_ray_segments used to *replace* the stored segments, so a scene
with two RaySegmentsItem bundles (e.g. on-axis + off-axis telescope beams)
silently dropped every bundle but the last — the drawn rays then never
matched the physics (a Keplerian's on-axis beam never appeared to cross the
optical axis at its marked intermediate image).
"""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _make_viewer():
    pytest.importorskip("vispy")
    from raytracer.viz.gl.viewer import OpenGLViewer, RenderConfig

    try:
        viewer = OpenGLViewer(
            x_lims=(-5, 5), y_lims=(-5, 5), size=(200, 200),
            render_config=RenderConfig(ray_width=0.4, sigma_factor=0.4,
                                       auto_exposure=False, exposure=1.0),
        )
        viewer.set_current()
        viewer.on_draw(None)
    except Exception as error:
        pytest.skip(f"No usable OpenGL context: {error}")
    return viewer


def test_two_bundles_both_render():
    viewer = _make_viewer()
    horizontal = np.array([[[-4.0, 2.0], [4.0, 2.0]]])
    vertical = np.array([[[0.0, -4.0], [0.0, 4.0]]])
    viewer.draw_ray_segments(horizontal, colors=np.array([[1.0, 0.0, 0.0, 1.0]]))
    viewer.draw_ray_segments(vertical, colors=np.array([[0.0, 1.0, 0.0, 1.0]]))
    viewer.on_draw(None)

    accum = viewer.read_accumulation()
    red = accum[..., 0].max()
    green = accum[..., 1].max()
    assert red > 0.0, "first bundle was dropped by the second draw call"
    assert green > 0.0, "second bundle did not render"

    viewer.clear_rays()
    viewer.on_draw(None)
    assert viewer.read_accumulation()[..., :3].max() == 0.0
    viewer.close()
