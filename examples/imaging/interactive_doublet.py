"""Interactive relay: drag the source/lens/sensor, tune parameters live.

A biconvex singlet images a point source onto a sensor; the light is a
meridional fan, retraced non-sequentially after every edit.

Controls:
  drag cyan handles  move the source, the lens vertex, or the sensor ends
  Tab                cycle editable parameters (shown in the window title)
  Left/Right         adjust the active parameter (Shift = coarse)
  mouse drag/wheel   pan / zoom (away from handles)

Usage:
    python -m examples.imaging.interactive_doublet
    python -m examples.imaging.interactive_doublet --save out.png   # the opening view, no window
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer import OpticalSystem
from raytracer.elements import Lens, Sensor
from raytracer.geometry.apertures import Rectangle
from raytracer.geometry.surfaces import Sphere
from raytracer.materials import ConstantIndex
from raytracer.sources import PointSource
from raytracer.viz import MarkerItem, show
from raytracer.viz.interactive import InteractiveSession

VIEW = dict(x_lims=(-110.0, 200.0), y_lims=(-60.0, 60.0), size=(1300, 700))


def main() -> None:
    lens = Lens("lens", front=Sphere(60.0), back=Sphere(-60.0), thickness=9.0,
                material=ConstantIndex("N1.5168", 1.5168), semi_diameter=16.0)
    screen = Sensor("screen", aperture=Rectangle(30.0, 30.0), z=171.4)
    source = PointSource((0.0, 0.0, -90.0), half_angle_deg=9.0, rays=201, fan="y")
    session = InteractiveSession(sources=[source], system=OpticalSystem([lens, screen]), max_interactions=5)

    if "--save" not in sys.argv:  # pragma: no cover - interactive
        session.run(**VIEW)
        return
    out = Path(sys.argv[sys.argv.index("--save") + 1])
    drawing = session.retrace()
    handles = np.array([h.get() for _, h in session.all_handles()])
    drawing.add(MarkerItem(handles, color="cyan", size=9.0))
    show(drawing, backend="gl", interactive=False, **VIEW).save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
