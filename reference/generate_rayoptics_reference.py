"""Regenerate an independent Cooke-triplet trace reference with RayOptics 0.9.8.

Run in an isolated environment with rayoptics==0.9.8 and opticalglass==1.2.0:
    PYTHONPATH=. python reference/generate_rayoptics_reference.py

Only the prescription and material indices come from this package. Intersection,
refraction, all paths and outgoing directions are evaluated by RayOptics. This
cross-check tests the tracer, not the accuracy of the shared material catalog.
RayOptics' equally-inclined-chord OPD is not an absolute OPL: the latter is
reconstructed from its signed geometric segment distances and gap indices.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import opticalglass
import rayoptics
from rayoptics.optical.opticalmodel import OpticalModel
from rayoptics.raytr.raytrace import trace

from raytracer.design import OpticalSystem

ROOT = Path(__file__).resolve().parents[1]
PRESCRIPTION = ROOT / "data/optical_systems/photographic/cooke_triplet_prescription.csv"
DESTINATION = ROOT / "tests/reference/rayoptics_cooke.json"


def main():
    cases = []
    for wavelength_um in (0.48613, 0.58756, 0.65627):
        system = OpticalSystem.from_prescription(PRESCRIPTION).at_wavelength(wavelength_um)
        model = OpticalModel()
        model.radius_mode = True
        model.optical_spec.spectral_region.wavelengths = [wavelength_um * 1000]
        sequence = model.seq_model
        sequence.gaps[0].thi = 100.0
        for row, index in zip(system.rows, system.n_after):
            sequence.add_surface([row.radius, row.thickness, float(index)])
        model.update_model()
        vertices = np.r_[-100.0, system.vertices, system.image_z]
        gap_indices = np.r_[1.0, system.n_after]
        for field_x, field_y in ((0.0, 0.0), (0.0, 10.0), (7.0, 7.0)):
            origin = np.array([field_x, field_y, -100.0])
            for pupil_x, pupil_y in ((0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)):
                direction = np.array([pupil_x - field_x, pupil_y - field_y, 100.0])
                direction /= np.linalg.norm(direction)
                segments, _, _ = trace(
                    sequence,
                    np.array([field_x, field_y, 0.0]),
                    direction,
                    wavelength_um * 1000,
                )
                paths = np.array([segment[0] for segment in segments])
                paths[:, 2] += vertices
                optical_length = sum(
                    float(segment[2]) * index for segment, index in zip(segments[:-1], gap_indices)
                )
                cases.append(
                    {
                        "wavelength_um": wavelength_um,
                        "indices_after": system.n_after.tolist(),
                        "origin_mm": origin.tolist(),
                        "direction": direction.tolist(),
                        "paths_mm": paths.tolist(),
                        "image_mm": paths[-1].tolist(),
                        "outgoing_direction": segments[-1][1].tolist(),
                        "opl_mm": optical_length,
                    }
                )
    payload = {
        "reference": "RayOptics",
        "version": rayoptics.__version__,
        "opticalglass_version": opticalglass.__version__,
        "prescription_sha256": hashlib.sha256(PRESCRIPTION.read_bytes()).hexdigest(),
        "description": "45 rays: 3 wavelengths, 3 finite fields, 5 launch directions; apertures disabled",
        "conventions": "mm; object_z=-100; signed paths; dummy stop may require a backward transfer",
        "cases": cases,
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(cases)} independent rays with RayOptics {rayoptics.__version__}")


if __name__ == "__main__":
    main()
