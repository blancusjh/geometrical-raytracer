"""Where the light focuses (``raytracer.viz.sag_drawing.beam_foci_from_paths``)."""

import numpy as np
import pytest

from raytracer.design import StigmaticTrain
from raytracer.propagation.fields import trace_from_object
from raytracer.propagation.sequential import SequentialTracer
from raytracer.viz.sag_drawing import beam_foci_from_paths


def relay_train():
    """Two stigmatic singlets in relay: the first images at z = 60 in the
    air gap between the lenses, the second re-images at z = 180."""

    return StigmaticTrain(
        media=(1.0, 1.5, 1.0, 1.5, 1.0),
        vertices=(0.0, 8.0, 100.0, 108.0),
        conjugates=(-60.0, 300.0, 60.0, 400.0, 180.0),
        semidiameter=6.0,
    )


def trace_fan(train, *, height=0.0, na=0.08, rays=7):
    system = train.to_system()
    tracer = SequentialTracer(system)
    chief = (0.0 - height) / (0.0 - system.object_z)
    paths = []
    for s in np.linspace(-na, na, rays):
        r = trace_from_object(tracer, (0.0, height), (0.0, chief + s),
                              keep_path=True)
        if r.ok:
            paths.append(r.path)
    return np.stack(paths)


def test_relay_reports_intermediate_and_final_focus():
    foci = beam_foci_from_paths(trace_fan(relay_train()))
    finals = [f for f in foci if f[4]]
    mids = [f for f in foci if not f[4]]
    assert len(finals) == 1 and finals[0][0] == pytest.approx(180.0, abs=1e-6)
    assert len(mids) == 1 and mids[0][0] == pytest.approx(60.0, abs=1e-6)
    assert mids[0][2] < 1e-9  # the stigmatic pinch is exact


def test_collimated_beam_reports_no_focus():
    """Parallel rays never converge: an afocal exit draws nothing."""

    paths = np.zeros((5, 3, 3))
    for k, h in enumerate(np.linspace(-1.0, 1.0, 5)):
        paths[k, :, 1] = h
        paths[k, :, 2] = (0.0, 10.0, 20.0)
    assert beam_foci_from_paths(paths) == []


def test_draw_system_marks_the_focus_by_default():
    """The relay's intermediate image (z = 60) and final image (z = 180)
    both get a focus marker — white face, field-colored edge."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from raytracer.viz.bodies import draw_system

    fig, ax = plt.subplots()
    try:
        draw_system(ax, relay_train(), fields=(0.0,), na=0.08)
        markers = [line for line in ax.lines
                   if line.get_marker() == "o"
                   and line.get_markerfacecolor() == "white"]
        z_marked = sorted(float(line.get_xdata()[0]) for line in markers)
        assert z_marked == pytest.approx([60.0, 180.0], abs=1e-6)
    finally:
        plt.close(fig)


def test_offframe_final_focus_is_noted():
    """Cropping the frame before the image plane must not silently drop
    the focus: a corner note states where the image is."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from raytracer.viz.bodies import draw_system

    fig, ax = plt.subplots()
    try:
        draw_system(ax, relay_train(), fields=(0.0,), na=0.08,
                    xlim=(-10.0, 130.0))
        assert any("image at z = 180" in t.get_text() for t in ax.texts)
        # the intermediate focus is inside the frame and still marked
        markers = [line for line in ax.lines
                   if line.get_marker() == "o"
                   and line.get_markerfacecolor() == "white"]
        assert [float(line.get_xdata()[0]) for line in markers] == (
            pytest.approx([60.0], abs=1e-6)
        )
    finally:
        plt.close(fig)
