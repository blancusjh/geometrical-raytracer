"""Interactive doublet: drag the source/lens/screen, tune parameters live.

Controls:
  drag cyan handles  move the source, the lens vertex, or the screen ends
  Tab                cycle editable parameters (shown in the window title)
  Left/Right         adjust the active parameter (Shift = coarse)
  mouse drag/wheel   pan / zoom (away from handles)

Usage: python -m examples.imaging.interactive_doublet
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer.optics import Lens, PointSource, Screen
from raytracer.propagation import TraceConfig
from raytracer.viz.interactive import InteractiveSession


def main() -> None:
    lens = Lens.from_radii(
        r1=60.0, r2=-60.0, thickness=9.0, semidiameter=16.0, n=1.5168,
        vertex=(0.0, 0.0), name="lens",
    )
    screen = Screen([171.4, -15.0], [171.4, 15.0])
    source = PointSource(
        origin=np.array([-90.0, 0.0]),
        axis_direction=np.array([1.0, 0.0]),
        aperture=np.deg2rad(18.0),
        samples=201,
    )
    session = InteractiveSession(
        sources=[source],
        elements=[lens],
        screens=[screen],
        trace_config=TraceConfig(max_generations=5),
    )
    session.run(x_lims=(-110.0, 200.0), y_lims=(-60.0, 60.0), size=(1300, 700))


if __name__ == "__main__":
    main()
