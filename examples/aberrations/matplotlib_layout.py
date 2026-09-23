"""Reproduce the existing Matplotlib layout with filled lenses and complete rays."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.aberrations.chromatic_sensitivity import doublet
from raytracer.propagation import FieldPoint, PupilSampling, SequentialTracer, trace_pupil
from raytracer.viz.geometric import layout_3d_figure

ROOT = Path(__file__).resolve().parents[2]


def main():
    system = doublet()
    paths, colors = [], []
    for field, color in [(0, "#b36b20"), (3, "#168277"), (6, "#a34c68")]:
        pupil = trace_pupil(
            SequentialTracer(system),
            FieldPoint.angle(y_deg=field),
            sampling=PupilSampling(kind="gauss", radial=2, azimuth=8),
            keep_paths=True,
        )
        if not pupil.valid.all():
            raise RuntimeError("The demonstration requires all rays to reach the screen")
        paths.extend(pupil.batch.paths)
        colors.extend([color] * len(pupil.batch.paths))
        residual = []
        for index, row in enumerate(system.rows):
            points = system.surface_frame(index).to_local(pupil.batch.paths[:, index + 1])
            residual.extend(
                abs(points[:, 2] - row.profile.sag(np.linalg.norm(points[:, :2], axis=1)))
            )
        print(
            f"Campo {field}°: {pupil.valid.sum()} rayos; residuo superficial {max(residual):.3e} mm"
        )
    fig = layout_3d_figure(
        system,
        paths=np.asarray(paths),
        ray_colors=colors,
        title="Doblete N-BK7 / N-F2 · campos 0°, 3° y 6° · 587,56 nm",
    )
    fig.savefig(ROOT / "docs/img/matplotlib_layout.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
