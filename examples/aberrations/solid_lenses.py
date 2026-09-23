"""Transparent solid volumes and real ray paths; optional native interaction.

Run: python -m examples.aberrations.solid_lenses [--interactive]
Requires: pip install 'raytracer[solid]'
"""

import argparse
from pathlib import Path

import numpy as np

from examples.aberrations.chromatic_sensitivity import doublet
from raytracer.design import OpticalSystem
from raytracer.propagation import FieldPoint, PupilSampling, SequentialTracer, trace_pupil
from raytracer.propagation.sequential import TraceStatus
from raytracer.viz import SolidViewer

ROOT = Path(__file__).resolve().parents[2]


def scene(system, *, title, fields=(0, 3, 6), detail=False):
    viewer = SolidViewer(system, title=title)
    colors = [(1, 0.68, 0.23), (0.40, 0.91, 0.77), (0.95, 0.48, 0.59)]
    for field, color in zip(fields, colors):
        pupil = trace_pupil(
            SequentialTracer(system),
            FieldPoint.angle(y_deg=field),
            sampling=PupilSampling(kind="fan_t", n=13),
            keep_paths=True,
        )
        if not np.isin(pupil.batch.status, [TraceStatus.OK, TraceStatus.VIGNETTED]).all():
            raise RuntimeError("Unexpected ray failure in solid-view example")
        viewer.add_paths(
            pupil.batch.paths, color=color, label=f"Campo {field} grados", detail=detail
        )
    return viewer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    cooke = OpticalSystem.from_prescription(
        ROOT / "data/optical_systems/photographic/cooke_triplet_prescription.csv"
    )
    # Put the stop in the actual air gap, clear of the preceding curved face.
    # The historical CSV puts it at that face's vertex, requiring virtual legs.
    cooke.set_thickness(3, 2.0)
    cooke.set_thickness(4, 4.0)
    for system, name, title, detail in [
        (doublet(), "doublet_solid", "Doblete cementado / N-BK7 + N-F2", True),
        (cooke, "cooke_solid", "Triplete Cooke / cuerpos opticos y trazado", False),
        (cooke, "cooke_solid_detail", "Triplete Cooke / detalle de las lentes", True),
    ]:
        viewer = scene(system, title=title, detail=detail)
        print(name, viewer.save(ROOT / f"docs/img/{name}.png"))
        if args.interactive:
            viewer.show()
        viewer.close()


if __name__ == "__main__":
    main()
