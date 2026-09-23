"""Independent RayOptics 0.9.8 Seidel and exact field reference.

Run with the isolated reference environment. This generator imports no
raytracer code: prescriptions, constant indices and field conventions are
explicit inputs. It uses RayOptics paraxial propagation and third-order
sums, plus independently aimed exact rays for differential field focus.
"""

import json
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from rayoptics.elem.profiles import EvenPolynomial
from rayoptics.optical.opticalmodel import OpticalModel
from rayoptics.parax.firstorder import ParaxData, paraxial_trace
from rayoptics.parax.thirdorder import compute_third_order
from rayoptics.raytr.raytrace import trace
from scipy.optimize import root

ROOT = Path(__file__).resolve().parents[1]


def cases():
    base = [dict(radius=50, thickness=5, index=1.5), dict(radius=-50, thickness=50, index=1.0)]
    return [
        dict(name="singlet_infinite", rows=base, object_z=None, field=3.0, stop=0, radius=5.0),
        dict(
            name="singlet_finite_rear_stop",
            rows=base,
            object_z=-120.0,
            field=8.0,
            stop=1,
            radius=4.0,
        ),
        dict(
            name="asphere_finite_rear_stop",
            rows=[
                dict(radius=40, thickness=6, index=1.6, conic=-0.7, a4=2e-6),
                dict(radius=-60, thickness=60, index=1.0, conic=0.2, a4=-1e-6),
            ],
            object_z=-150.0,
            field=10.0,
            stop=1,
            radius=4.0,
        ),
        *[
            dict(
                name=name,
                rows=[dict(radius=-100.0, thickness=-50.0, mirror=True, conic=conic)],
                object_z=None,
                field=3.0,
                stop=0,
                radius=5.0,
            )
            for name, conic in [("spherical_mirror", 0.0), ("parabolic_mirror", -1.0)]
        ],
    ]


def generate(case):
    model = OpticalModel()
    model.radius_mode = True
    model.optical_spec.spectral_region.wavelengths = [550.0]
    seq = model.seq_model
    seq.gaps[0].thi = 100.0
    for row in case["rows"]:
        medium = "REFL" if row.get("mirror") else row["index"]
        seq.add_surface([row["radius"], row["thickness"], medium])
        # EvenPolynomial coefficients start at h^2: zero a2, then a4.
        seq.ifcs[-2].profile = EvenPolynomial(
            c=1 / row["radius"], cc=row.get("conic", 0.0), coefs=[0.0, row.get("a4", 0.0)]
        )
    model.update_model()
    basis_y, basis_u = paraxial_trace(seq.path(550.0), 1, [1.0, 0.0], [0.0, 1.0])
    stop = case["stop"] + 1
    stop_row = np.array([basis_y[stop][0], basis_u[stop][0]])
    if case["object_z"] is None:
        variable = np.array([1.0, 0.0])
        fixed = np.array([0.0, np.tan(np.deg2rad(case["field"]))])
    else:
        variable = np.array([-case["object_z"], 1.0])
        fixed = np.array([case["field"], 0.0])
    marginal = variable * case["radius"] / (stop_row @ variable)
    chief = fixed - variable * (stop_row @ fixed) / (stop_row @ variable)
    axial, principal = paraxial_trace(seq.path(550.0), 1, marginal.tolist(), chief.tolist())
    invariant = marginal[0] * chief[1] - chief[0] * marginal[1]
    model["analysis_results"]["parax_data"] = ParaxData(
        axial, principal, SimpleNamespace(opt_inv=invariant)
    )
    ledger = compute_third_order(model)
    contributions = []
    for index in range(1, len(case["rows"]) + 1):
        values = ledger.loc[str(index)].to_numpy()
        if str(index) + ".asp" in ledger.index:
            values = values + ledger.loc[str(index) + ".asp"].to_numpy()
        contributions.append(values.tolist())
    z_image = sum(row["thickness"] for row in case["rows"])
    gaussian_z = z_image - axial[-1][0] / axial[-1][1]
    image_height = principal[-1][0] + (gaussian_z - z_image) * principal[-1][1]
    # Set the last physical gap to the Gaussian image for exact-ray comparisons.
    seq.gaps[-1].thi += gaussian_z - z_image
    model.update_model()

    def ray(parameters):
        if case["object_z"] is None:
            direction = np.array([0.0, np.tan(np.deg2rad(case["field"])), 1.0])
            origin = np.r_[parameters, 0.0] - 100 * direction
        else:
            direction = np.r_[parameters, 1.0]
            origin = np.array([0.0, case["field"], case["object_z"]])
            origin += (-100 - case["object_z"]) * direction
        origin[2] = 0.0  # RayOptics object surface is local z=0 at system z=-100.
        direction /= np.linalg.norm(direction)
        return trace(seq, origin, direction, 550.0)[0]

    chief_parameters = [0.0, chief[0]] if case["object_z"] is None else [0.0, chief[1]]

    def aimed(target):
        solution = root(
            lambda p: np.asarray(ray(p)[stop][0])[:2] - target, chief_parameters, tol=1e-11
        )
        residual = np.linalg.norm(np.asarray(ray(solution.x)[stop][0])[:2] - target)
        if not np.isfinite(residual) or residual > 2e-10:
            raise RuntimeError(f"Reference aiming failed: {residual}")
        result = ray(solution.x)[-1]
        return np.asarray(result[0])[:2], np.asarray(result[1])[:2] / result[1][2]

    point, _ = aimed(np.zeros(2))
    foci = []
    step = 5e-4
    for axis in (np.array([0.0, 1.0]), np.array([1.0, 0.0])):
        minus, plus = [aimed(sign * step * axis) for sign in (-1, 1)]
        foci.append(float(-((plus[0] - minus[0]) @ axis) / ((plus[1] - minus[1]) @ axis)))
    return {
        **case,
        "surface_sums_mm": contributions,
        "sums_mm": ledger.loc["sum"].tolist(),
        "marginal_first_vertex": marginal.tolist(),
        "chief_first_vertex": chief.tolist(),
        "invariant_mm": float(invariant),
        "gaussian_z_mm": float(gaussian_z),
        "gaussian_height_mm": float(image_height),
        "chief_image_xy_mm": point.tolist(),
        "parabasal_step_mm": step,
        "parabasal_shifts_mm": foci,
    }


def main():
    payload = {
        "reference": "RayOptics",
        "version": version("rayoptics"),
        "description": "Five constant-index systems; independent paraxial rays, surface sums and exact parabasal traces.",
        "units": {"length": "mm", "angle": "deg"},
        "wavelength_um": 0.55,
        "field_convention": "object_z null: angular field with slope tan(theta); otherwise object height",
        "cases": [generate(case) for case in cases()],
    }
    path = ROOT / "tests/reference/rayoptics_third_order.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    for case in payload["cases"]:
        print(case["name"], case["sums_mm"], case["parabasal_shifts_mm"])


if __name__ == "__main__":
    main()
