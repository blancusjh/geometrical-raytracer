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


def _tree(starts_zy, ends_zy):
    """A RayTree of primary segments in the drawing plane (z, y), all ending on a face."""

    from raytracer.tracing import RayTree

    starts = np.array([[0.0, y, z] for z, y in starts_zy], dtype=float)
    ends = np.array([[0.0, y, z] for z, y in ends_zy], dtype=float)
    n = starts.shape[0]
    d = ends - starts
    return RayTree(
        start=starts,
        end=ends,
        direction=d / np.linalg.norm(d, axis=1, keepdims=True),
        power=np.ones(n),
        wavelength_um=np.full(n, np.nan),
        generation=np.ones(n, dtype=int),
        parent=np.full(n, -1),
        kind=np.zeros(n, dtype=int),
        face=["s"] * n,
        opl_end=np.zeros(n),
    )


def test_accumulation_is_linear_additive():
    viewer = _make_canvas()
    one = _tree([(-3.0, 0.0)], [(3.0, 0.0)])
    two = _tree([(-3.0, 0.0)] * 2, [(3.0, 0.0)] * 2)

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
    viewer.draw_rays(_tree([(-3.0, 0.001 * k) for k in range(200)], [(3.0, 0.001 * k) for k in range(200)]))
    _draw(viewer)

    accum = viewer.read_accumulation()[..., :3].max()
    assert accum > 10.0  # HDR buffer accumulated far beyond display range

    display = viewer.snapshot()[..., :3]
    # Tone mapping compresses to displayable range; auto-exposure pins the
    # 99th percentile near (but below) full scale.
    assert display.max() <= 255
    assert display.max() > 100  # bright, not black
    viewer.close()
