"""US 7,557,996 B2, Figure 3 / Tables 3 and 3A.

Exact sequential ray tracing of the rotationally symmetric catadioptric
projection objective, plus a scalar diffraction imaging demonstration.
All prescription dimensions are millimetres.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize


WAVELENGTH_MM = 193.368e-6
NA_IMAGE = 1.2
REDUCTION = 4.0


@dataclass
class Surface:
    number: int
    radius: float
    thickness: float
    material: str
    index: float
    semidiameter: float
    coefficients: tuple[float, ...] = ()
    vertex: float = 0.0

    @property
    def reflective(self) -> bool:
        return self.material == "REFL"


# radius, thickness, material after the surface, index, clear semidiameter
PRESCRIPTION = [
    (-404.542022, 36.232225, "SIO2", 1.56078570, 88.996),
    (-184.402723, 1.000090, "AIR", 1.0, 95.088),
    (674.592973, 51.347982, "SIO2", 1.56078570, 102.076),
    (-174.232362, 144.581202, "AIR", 1.0, 102.829),
    (-162.800929, 14.999049, "CAF2", 1.50185255, 53.189),
    (-1456.920767, 15.293417, "AIR", 1.0, 63.896),
    (-202.841004, -15.293417, "REFL", 1.0, 67.884),
    (-1456.920767, -14.999049, "CAF2", 1.50185255, 67.836),
    (-162.800929, -129.581898, "AIR", 1.0, 67.241),
    (-2538.490880, 174.873667, "REFL", 1.0, 89.519),
    (-391.507349, 46.632422, "SIO2", 1.56078570, 140.865),
    (-195.389639, 0.999852, "AIR", 1.0, 144.625),
    (1411.077988, 41.703225, "SIO2", 1.56078570, 160.805),
    (-529.305238, 0.999461, "AIR", 1.0, 162.519),
    (1458.778723, 34.033769, "SIO2", 1.56078570, 165.701),
    (-3002.433858, 1.088813, "AIR", 1.0, 166.002),
    (212.825771, 57.638065, "SIO2", 1.56078570, 165.361),
    (436.070633, 107.276235, "AIR", 1.0, 157.551),
    (807.065576, 13.824876, "SIO2", 1.56078570, 136.972),
    (290.022861, 105.550931, "AIR", 1.0, 129.323),
    (153.210365, 45.794134, "SIO2", 1.56078570, 102.069),
    (136.220631, 56.777310, "AIR", 1.0, 85.525),
    (-236.154610, 16.122149, "SIO2", 1.56078570, 84.150),
    (196.227428, 58.167782, "AIR", 1.0, 89.217),
    (-225.833560, 59.831340, "SIO2", 1.56078570, 97.239),
    (-193.684381, 3.215861, "AIR", 1.0, 118.689),
    (1270.382760, 13.092290, "SIO2", 1.56078570, 138.308),
    (444.089768, 15.430530, "AIR", 1.0, 145.054),
    (798.722954, 70.490347, "SIO2", 1.56078570, 148.503),
    (-372.865243, 1.829734, "AIR", 1.0, 155.186),
    (1055.533545, 39.622524, "SIO2", 1.56078570, 165.276),
    (-843.038518, 36.110674, "AIR", 1.0, 165.999),
    (397.601006, 52.148862, "SIO2", 1.56078570, 165.961),
    (-2316.548771, -4.213465, "AIR", 1.0, 164.343),
    (0.0, 0.0, "AIR", 1.0, 163.851),
    (0.0, 24.329363, "AIR", 1.0, 163.851),
    (-584.913340, 13.012880, "SIO2", 1.56078570, 163.706),
    (-1175.679472, 0.999882, "AIR", 1.0, 163.968),
    (468.096624, 53.626064, "SIO2", 1.56078570, 161.375),
    (-861.698483, 0.999930, "AIR", 1.0, 159.675),
    (199.043165, 50.143874, "SIO2", 1.56078570, 132.604),
    (1068.213664, 0.999723, "AIR", 1.0, 126.983),
    (159.145519, 28.472240, "SIO2", 1.56078570, 103.283),
    (185.472842, 1.000460, "AIR", 1.0, 90.240),
    (118.935626, 36.986741, "CAF2", 1.50185255, 82.041),
    (183.063185, 1.167554, "AIR", 1.0, 65.440),
    (108.701433, 56.721488, "HIINDEX1", 1.70196985, 58.621),
    (0.0, 1.996512, "HIINDEX2", 1.59667693, 18.135),
]


ASPHERES = {
    2: (-2.722625e-9, 7.584994e-13, 9.846612e-17, 6.877818e-21, 1.011674e-24, -1.051032e-29),
    5: (1.960629e-8, 7.765801e-13, 1.917740e-17, 2.910700e-22, 1.224735e-25, -3.426250e-30),
    9: (1.960629e-8, 7.765801e-13, 1.917740e-17, 2.910700e-22, 1.224735e-25, -3.426250e-30),
    10: (-2.009517e-9, -9.097265e-14, -8.260726e-18, 5.085547e-23, -1.203685e-26, -9.728791e-31),
    13: (4.688860e-12, -1.731249e-13, -6.240777e-19, 4.036403e-23, -1.100969e-27, 5.706583e-33),
    18: (1.982111e-8, -8.423501e-14, -7.592975e-19, 1.931064e-22, -5.204262e-27, 8.863320e-32),
    19: (1.777327e-8, 3.329332e-13, 2.038481e-20, 7.263940e-22, -3.379214e-26, 6.878653e-31),
    23: (-1.163364e-7, -9.570513e-13, 4.236584e-17, -7.733258e-22, 2.136062e-26, 1.705507e-30),
    27: (-7.203585e-9, 1.049118e-13, -7.379754e-19, 7.196196e-23, -9.945374e-28, 4.329096e-32),
    42: (1.122901e-8, -3.595965e-14, -2.518839e-18, 4.313506e-22, -1.564968e-26, 3.223390e-31),
    44: (1.449926e-8, 5.832083e-13, 6.975570e-17, 3.546013e-21, -1.283131e-26, 1.520482e-29),
    46: (6.313252e-8, 1.940331e-12, -2.373686e-16, -1.406348e-21, -1.469813e-25, 9.584030e-29),
}


def make_surfaces() -> list[Surface]:
    surfaces: list[Surface] = []
    z = 0.0
    for number, row in enumerate(PRESCRIPTION, 1):
        radius, thickness, material, index, semidiameter = row
        surfaces.append(Surface(number, radius, thickness, material, index,
                                semidiameter, ASPHERES.get(number, ()), z))
        z += thickness
    return surfaces


SURFACES = make_surfaces()
IMAGE_Z = SURFACES[-1].vertex + SURFACES[-1].thickness


def sag_and_slope(surface: Surface, radius: np.ndarray | float):
    h = np.asarray(radius, dtype=float)
    if surface.radius == 0:
        base = np.zeros_like(h)
        slope = np.zeros_like(h)
    else:
        c = 1.0 / surface.radius
        q = np.sqrt(np.maximum(1.0 - c * c * h * h, 1e-30))
        base = c * h * h / (1.0 + q)
        slope = c * h / q
    poly = np.zeros_like(h)
    dpoly = np.zeros_like(h)
    for j, coefficient in enumerate(surface.coefficients, 2):
        poly += coefficient * h ** (2 * j)
        dpoly += 2 * j * coefficient * h ** (2 * j - 1)
    return base + poly, slope + dpoly


def intersect(point: np.ndarray, direction: np.ndarray, surface: Surface):
    if abs(direction[2]) < 1e-14:
        raise RuntimeError("Ray is parallel to a vertex plane")
    t = (surface.vertex - point[2]) / direction[2]
    for _ in range(15):
        p = point + t * direction
        h = np.hypot(p[0], p[1])
        sag, slope = sag_and_slope(surface, h)
        f = p[2] - surface.vertex - float(sag)
        radial_dot = 0.0 if h == 0 else (p[0] * direction[0] + p[1] * direction[1]) / h
        derivative = direction[2] - float(slope) * radial_dot
        step = f / derivative
        t -= step
        if abs(step) < 2e-11:
            break
    p = point + t * direction
    h = np.hypot(p[0], p[1])
    if h > surface.semidiameter + 1e-7:
        raise RuntimeError(f"Ray clipped at surface {surface.number}")
    _, slope = sag_and_slope(surface, h)
    normal = np.array([0.0, 0.0, 1.0]) if h == 0 else np.array(
        [-float(slope) * p[0] / h, -float(slope) * p[1] / h, 1.0])
    normal /= np.linalg.norm(normal)
    return p, normal


def redirect(direction: np.ndarray, normal: np.ndarray, n_before: float,
             n_after: float, reflective: bool):
    normal = normal.copy()
    if np.dot(direction, normal) > 0:
        normal *= -1
    cos_i = -np.dot(direction, normal)
    if reflective:
        out = direction + 2 * cos_i * normal
    else:
        eta = n_before / n_after
        radicand = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
        if radicand < 0:
            raise RuntimeError("Total internal reflection")
        out = eta * direction + (eta * cos_i - np.sqrt(radicand)) * normal
    return out / np.linalg.norm(out)


def trace(point: np.ndarray, direction: np.ndarray, keep_path: bool = False):
    point = np.asarray(point, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    n_current = 1.0
    path = [point.copy()]
    optical_path = 0.0
    for surface in SURFACES:
        hit, normal = intersect(point, direction, surface)
        optical_path += n_current * np.linalg.norm(hit - point)
        path.append(hit.copy())
        n_after = n_current if surface.reflective else surface.index
        direction = redirect(direction, normal, n_current, n_after, surface.reflective)
        n_current = n_after
        point = hit + direction * 1e-8
    t = (IMAGE_Z - point[2]) / direction[2]
    image = point + t * direction
    optical_path += n_current * abs(t)
    path.append(image.copy())
    result = {"image": image, "direction": direction, "opl": optical_path}
    if keep_path:
        result["path"] = np.asarray(path)
    return result


def direction_from_slopes(sx: float, sy: float):
    return np.array([sx, sy, 1.0]) / np.sqrt(1.0 + sx * sx + sy * sy)


def find_object_plane():
    """Numerically recover the omitted object-space conjugate distance."""
    eps = 1e-4
    start_z = -1e-6
    base = np.array([0.0, 0.0, start_z])
    y_from_height = trace(base + [0, eps, 0], direction_from_slopes(0, 0))["image"][1]
    y_from_angle = trace(base, direction_from_slopes(0, eps))["image"][1]
    a = y_from_height / eps
    b = y_from_angle / eps
    distance = -b / a
    return -distance, a


OBJECT_Z, PARAXIAL_MAGNIFICATION = find_object_plane()


def ray_from_object(object_xy, slopes, keep_path=False):
    point = np.array([object_xy[0], object_xy[1], OBJECT_Z])
    return trace(point, direction_from_slopes(slopes[0], slopes[1]), keep_path)


def chief_slope(object_y: float):
    """Chief ray: force the ray through the centre of the aperture stop."""
    stop_index = 34  # patent surface 35

    def at_stop(sy):
        result = ray_from_object((0, object_y), (0, float(sy)), True)
        return result["path"][stop_index + 1, 1]

    samples = np.linspace(-0.08, 0.08, 65)
    valid = []
    for value in samples:
        try:
            valid.append((value, at_stop(value)))
        except RuntimeError:
            continue
    for (a, fa), (b, fb) in zip(valid, valid[1:]):
        if fa == 0:
            return a
        if fa * fb < 0:
            return optimize.brentq(at_stop, a, b)
    raise RuntimeError(f"No unvignetted chief ray found for object y={object_y}")


def pupil_rays(object_y=68.0, radial_samples=9, azimuth_samples=32):
    """Generate object-space slopes and retain all unvignetted rays."""
    sy0 = chief_slope(object_y)
    # NA transforms approximately by the 4x reduction.
    max_sine = NA_IMAGE / REDUCTION
    max_slope = max_sine / np.sqrt(1.0 - max_sine * max_sine)
    rays = []
    for radial in np.linspace(0, 1, radial_samples):
        count = 1 if radial == 0 else azimuth_samples
        for azimuth in np.linspace(0, 2 * np.pi, count, endpoint=False):
            sx = radial * max_slope * np.cos(azimuth)
            sy = sy0 + radial * max_slope * np.sin(azimuth)
            try:
                rays.append((radial, azimuth, ray_from_object((0, object_y), (sx, sy))))
            except RuntimeError:
                pass
    return rays


def physical_lenses():
    """Return unique entrance/exit surface pairs for each solid lens."""
    solid_materials = {"SIO2", "CAF2", "HIINDEX1"}
    pairs = []
    signatures = set()
    for first, second in zip(SURFACES[:-1], SURFACES[1:]):
        if first.material not in solid_materials or second.material == first.material:
            continue
        endpoints = sorted([
            (round(first.vertex, 6), round(first.radius, 6)),
            (round(second.vertex, 6), round(second.radius, 6)),
        ])
        signature = (first.material, tuple(endpoints))
        if signature in signatures:
            continue
        signatures.add(signature)
        pairs.append((first, second, first.material))
    return pairs


def plot_layout(output: Path):
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {"SIO2": "#bde0fe", "CAF2": "#c7f9cc", "HIINDEX1": "#ffe29a"}
    edges = {"SIO2": "#2878b5", "CAF2": "#278b65", "HIINDEX1": "#c47f00"}

    # Filled, closed lens bodies: both optical faces plus the two rim edges.
    for first, second, material in physical_lenses():
        clear = min(first.semidiameter, second.semidiameter)
        h = np.linspace(-clear, clear, 240)
        z1 = first.vertex + sag_and_slope(first, np.abs(h))[0]
        z2 = second.vertex + sag_and_slope(second, np.abs(h))[0]
        polygon_z = np.r_[z1, z2[::-1]]
        polygon_h = np.r_[h, h[::-1]]
        ax.fill(polygon_z, polygon_h, facecolor=colors[material],
                edgecolor=edges[material], lw=1.0, alpha=0.58, zorder=1)

    # The two reflective faces.
    for surface, label in ((SURFACES[6], "M1"), (SURFACES[9], "M2")):
        h = np.linspace(-surface.semidiameter, surface.semidiameter, 240)
        z = surface.vertex + sag_and_slope(surface, np.abs(h))[0]
        ax.plot(z, h, color="#202020", lw=3.0, zorder=2)
        ax.text(float(np.mean(z)), surface.semidiameter + 8, label,
                ha="center", fontsize=9)

    # High-index immersion medium after the planar final solid surface.
    last = SURFACES[-1]
    ax.fill_betweenx([-last.semidiameter, last.semidiameter], last.vertex,
                     IMAGE_Z, color="#f6bd60", alpha=0.24, zorder=0)

    # The active annular strip extends to image height 17 mm (object 68 mm).
    # Keep the plotted samples just inside that hard clear-aperture boundary.
    object_fields = [56.0, 62.0, 67.0]
    ray_colors = ["#d62728", "#2ca02c", "#1f77b4"]
    for field, color in zip(object_fields, ray_colors):
        sy0 = chief_slope(field)
        for delta in np.linspace(-0.29, 0.29, 7):
            try:
                path = ray_from_object((0, field), (0, sy0 + delta), True)["path"]
                ax.plot(path[:, 2], path[:, 1], color=color, lw=0.65,
                        alpha=0.72, zorder=3)
            except RuntimeError:
                continue
    ax.axvline(OBJECT_Z, color="black", ls="--", lw=1, label="object plane")
    ax.axvline(IMAGE_Z, color="black", ls="--", lw=1, label="image plane")
    ax.axvline(SURFACES[34].vertex, color="#555555", ls=":", lw=1,
               label="aperture stop")
    ax.text(OBJECT_Z, 105, "OS", ha="center", fontsize=10)
    ax.text(IMAGE_Z, 105, "IS", ha="center", fontsize=10)
    ax.text(SURFACES[34].vertex, 175, "AS", ha="center", fontsize=9)
    ax.axhline(0, color="#aaaaaa", lw=0.6)
    ax.set(xlabel="Global z (mm)", ylabel="Meridional y (mm)",
           title="US7557996 — Fig. 3 / Table 3 exact sequential layout")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def export_prescription_csv(output: Path):
    fields = [
        "surface", "surface_type", "radius_mm", "thickness_to_next_mm",
        "material_after", "index_193.368_nm", "clear_semidiameter_mm",
        "aspheric", "K", "C1_mm^-3", "C2_mm^-5", "C3_mm^-7",
        "C4_mm^-9", "C5_mm^-11", "C6_mm^-13", "vertex_z_mm",
    ]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for surface in SURFACES:
            if surface.reflective:
                kind = "mirror"
            elif surface.number == 35:
                kind = "aperture_stop"
            elif surface.radius == 0:
                kind = "plane"
            else:
                kind = "refractive"
            coefficients = list(surface.coefficients) + [""] * 6
            writer.writerow({
                "surface": surface.number,
                "surface_type": kind,
                "radius_mm": f"{surface.radius:.9f}",
                "thickness_to_next_mm": f"{surface.thickness:.9f}",
                "material_after": surface.material,
                "index_193.368_nm": f"{surface.index:.8f}",
                "clear_semidiameter_mm": f"{surface.semidiameter:.3f}",
                "aspheric": bool(surface.coefficients),
                "K": 0 if surface.coefficients else "",
                "C1_mm^-3": coefficients[0],
                "C2_mm^-5": coefficients[1],
                "C3_mm^-7": coefficients[2],
                "C4_mm^-9": coefficients[3],
                "C5_mm^-11": coefficients[4],
                "C6_mm^-13": coefficients[5],
                "vertex_z_mm": f"{surface.vertex:.9f}",
            })


def plot_spots(output: Path):
    fields = [56.0, 62.0, 67.0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for field, ax in zip(fields, axes):
        rays = pupil_rays(field, 10, 48)
        points = np.array([ray[2]["image"][:2] for ray in rays])
        radial = np.array([ray[0] for ray in rays])
        weights = np.maximum(radial, 0.5 / 9)
        weights /= weights.sum()
        centre = np.sum(points * weights[:, None], axis=0)
        relative_um = (points - centre) * 1e3
        ax.scatter(relative_um[:, 0], relative_um[:, 1], s=4, alpha=0.55)
        rms = np.sqrt(np.sum(weights * np.sum(relative_um**2, axis=1)))
        ax.set_title(f"Object y={field:.0f} mm\narea-weighted RMS={rms:.3f} µm")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_xlabel("Δx (µm)")
    axes[0].set_ylabel("Δy (µm)")
    fig.suptitle("Geometric spot diagrams — unvignetted rays")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _weighted_mean(values, weights):
    return np.sum(values * weights) / np.sum(weights)


def _weighted_quantile(values, weights, quantile):
    order = np.argsort(values)
    values = np.asarray(values)[order]
    weights = np.asarray(weights)[order]
    cumulative = np.cumsum(weights) / np.sum(weights)
    return float(np.interp(quantile, cumulative, values))


def aberration_metrics(fields=(56.0, 62.0, 67.0)):
    """Monochromatic transverse, distortion, focus, and astigmatism metrics."""
    results = []
    for field in fields:
        rays = pupil_rays(field, radial_samples=10, azimuth_samples=48)
        points = np.array([item[2]["image"][:2] for item in rays])
        directions = np.array([item[2]["direction"] for item in rays])
        radial = np.array([item[0] for item in rays])
        # Equal radial steps need an r weight to represent equal pupil area.
        weights = np.maximum(radial, 0.5 / 9)
        weights /= weights.sum()
        centroid = np.sum(points * weights[:, None], axis=0)
        relative = points - centroid
        radius_um = np.linalg.norm(relative, axis=1) * 1e3
        rms_x_um = np.sqrt(_weighted_mean((relative[:, 0] * 1e3) ** 2, weights))
        rms_y_um = np.sqrt(_weighted_mean((relative[:, 1] * 1e3) ** 2, weights))
        rms_radius_um = np.sqrt(_weighted_mean(radius_um ** 2, weights))
        ee80_radius_um = _weighted_quantile(radius_um, weights, 0.8)

        slopes = directions[:, :2] / directions[:, 2, None]
        mean_slopes = np.sum(slopes * weights[:, None], axis=0)
        slope_relative = slopes - mean_slopes

        def best_focus(axis):
            numerator = np.sum(weights * relative[:, axis] * slope_relative[:, axis])
            denominator = np.sum(weights * slope_relative[:, axis] ** 2)
            return -numerator / denominator

        sagittal_focus = best_focus(0)
        tangential_focus = best_focus(1)
        numerator = np.sum(weights[:, None] * relative * slope_relative)
        denominator = np.sum(weights[:, None] * slope_relative ** 2)
        best_image_focus = -numerator / denominator
        best_points = points + best_image_focus * slopes
        best_centroid = np.sum(best_points * weights[:, None], axis=0)
        best_relative_um = (best_points - best_centroid) * 1e3
        best_focus_rms_um = np.sqrt(
            np.sum(weights * np.sum(best_relative_um ** 2, axis=1)))
        ideal_y = field / REDUCTION
        results.append({
            "object_height_mm": field,
            "paraxial_image_height_mm": ideal_y,
            "centroid_image_height_mm": centroid[1],
            "distortion_um": (centroid[1] - ideal_y) * 1e3,
            "relative_distortion_ppm": (centroid[1] / ideal_y - 1) * 1e6,
            "rms_spot_radius_um": rms_radius_um,
            "rms_sagittal_um": rms_x_um,
            "rms_tangential_um": rms_y_um,
            "ee80_radius_um": ee80_radius_um,
            "best_focus_shift_mm": best_image_focus,
            "best_focus_rms_spot_um": best_focus_rms_um,
            "sagittal_focus_shift_mm": sagittal_focus,
            "tangential_focus_shift_mm": tangential_focus,
            "astigmatic_separation_mm": tangential_focus - sagittal_focus,
            "unvignetted_rays": len(rays),
        })
    return results


def export_aberration_csv(output: Path, metrics=None):
    metrics = aberration_metrics() if metrics is None else metrics
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)


def plot_aberrations(summary_output: Path, fan_output: Path, metrics=None):
    metrics = aberration_metrics() if metrics is None else metrics
    image_height = np.array([row["centroid_image_height_mm"] for row in metrics])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(image_height, [row["rms_spot_radius_um"] for row in metrics],
                    "o-", label="nominal plane")
    axes[0, 0].plot(image_height, [row["best_focus_rms_spot_um"] for row in metrics],
                    "s-", label="best focus")
    axes[0, 0].set(ylabel="RMS radius (µm)", title="Transverse aberration")
    axes[0, 0].legend()
    axes[0, 1].plot(image_height, [row["distortion_um"] for row in metrics], "o-")
    axes[0, 1].axhline(0, color="black", lw=0.7)
    axes[0, 1].set(ylabel="Centroid distortion (µm)", title="Distortion")
    axes[1, 0].plot(image_height, [row["sagittal_focus_shift_mm"] for row in metrics],
                    "o-", label="sagittal")
    axes[1, 0].plot(image_height, [row["tangential_focus_shift_mm"] for row in metrics],
                    "s-", label="tangential")
    axes[1, 0].axhline(0, color="black", lw=0.7)
    axes[1, 0].set(ylabel="Best-focus shift (mm)", title="Field curvature")
    axes[1, 0].legend()
    axes[1, 1].plot(image_height,
                    [row["astigmatic_separation_mm"] for row in metrics], "o-")
    axes[1, 1].axhline(0, color="black", lw=0.7)
    axes[1, 1].set(ylabel="T−S focus (mm)", title="Astigmatism")
    for ax in axes.flat:
        ax.set_xlabel("Image height (mm)")
        ax.grid(alpha=0.2)
    fig.suptitle("US7557996 monochromatic aberration measurements")
    fig.tight_layout()
    fig.savefig(summary_output, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    maximum_sine = NA_IMAGE / REDUCTION
    maximum_slope = maximum_sine / np.sqrt(1 - maximum_sine ** 2)
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    for field, color in zip((56.0, 62.0, 67.0), colors):
        chief = chief_slope(field)
        tangential = []
        sagittal = []
        for pupil in np.linspace(-1, 1, 81):
            try:
                ray_t = ray_from_object((0, field), (0, chief + pupil * maximum_slope))
                tangential.append((pupil, (ray_t["image"][1] - field / REDUCTION) * 1e3))
            except RuntimeError:
                pass
            try:
                ray_s = ray_from_object((0, field), (pupil * maximum_slope, chief))
                sagittal.append((pupil, ray_s["image"][0] * 1e3))
            except RuntimeError:
                pass
        tangential = np.asarray(tangential)
        sagittal = np.asarray(sagittal)
        axes[0].plot(tangential[:, 0], tangential[:, 1], color=color,
                     label=f"y={field:.0f} mm")
        axes[1].plot(sagittal[:, 0], sagittal[:, 1], color=color,
                     label=f"y={field:.0f} mm")
    axes[0].set(title="Tangential ray fan", ylabel="Δy′ (µm)")
    axes[1].set(title="Sagittal ray fan", ylabel="Δx′ (µm)")
    for ax in axes:
        ax.set_xlabel("Normalized pupil coordinate")
        ax.axhline(0, color="black", lw=0.7)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(fan_output, dpi=180)
    plt.close(fig)


ZERNIKE_NAMES = {
    (1, -1): "Y tilt", (1, 1): "X tilt",
    (2, -2): "Oblique astigmatism", (2, 0): "Defocus",
    (2, 2): "Vertical astigmatism",
    (3, -3): "Oblique trefoil", (3, -1): "Y coma",
    (3, 1): "X coma", (3, 3): "Vertical trefoil",
    (4, -4): "Oblique quadrafoil", (4, -2): "Secondary oblique astigmatism",
    (4, 0): "Primary spherical", (4, 2): "Secondary vertical astigmatism",
    (4, 4): "Vertical quadrafoil",
    (5, -1): "Secondary Y coma", (5, 1): "Secondary X coma",
    (6, 0): "Secondary spherical",
}


def zernike_modes(max_order=6):
    modes = []
    for n in range(1, max_order + 1):  # piston is not observable from ray slopes
        for m in range(-n, n + 1, 2):
            ansi = (n * (n + 2) + m) // 2
            modes.append((ansi, n, m, ZERNIKE_NAMES.get((n, m), f"Z({n},{m})")))
    return modes


def _zernike_radial(n, m, rho):
    m = abs(m)
    result = np.zeros_like(np.asarray(rho, dtype=float))
    for k in range((n - m) // 2 + 1):
        coefficient = (
            (-1) ** k * math.factorial(n - k)
            / (math.factorial(k)
               * math.factorial((n + m) // 2 - k)
               * math.factorial((n - m) // 2 - k))
        )
        result += coefficient * np.asarray(rho) ** (n - 2 * k)
    return result


def zernike(n, m, u, v):
    """ANSI/OSA RMS-normalized Zernike on the unit disk."""
    rho = np.hypot(u, v)
    theta = np.arctan2(v, u)
    normalization = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * (n + 1))
    angular = np.sin(abs(m) * theta) if m < 0 else np.cos(m * theta)
    return normalization * _zernike_radial(n, m, rho) * angular


def zernike_expansion(object_y=62.0, max_order=6):
    """Reconstruct wavefront coefficients from transverse ray aberrations.

    With normalized image-space optical-direction coordinates q, Hamilton's
    ray/wavefront relation is grad_q(W) = -NA * transverse_ray_aberration.
    Fitting the gradients avoids phase ambiguity through the folded path.
    """
    rays = pupil_rays(object_y, radial_samples=12, azimuth_samples=72)
    points = np.array([item[2]["image"][:2] for item in rays])
    directions = np.array([item[2]["direction"] for item in rays])
    radial = np.array([item[0] for item in rays])
    weights = np.maximum(radial, 0.5 / 11)
    weights /= weights.sum()
    chief_direction = ray_from_object(
        (0, object_y), (0, chief_slope(object_y)))["direction"]
    image_index = SURFACES[-1].index
    u = image_index * (directions[:, 0] - chief_direction[0]) / NA_IMAGE
    v = image_index * (directions[:, 1] - chief_direction[1]) / NA_IMAGE
    keep = np.hypot(u, v) <= 1.01
    u, v, points, weights = u[keep], v[keep], points[keep], weights[keep]
    weights /= weights.sum()
    centroid = np.sum(points * weights[:, None], axis=0)
    transverse = points - centroid

    modes = zernike_modes(max_order)
    step = 1e-5
    derivative_u = np.column_stack([
        (zernike(n, m, u + step, v) - zernike(n, m, u - step, v)) / (2 * step)
        for _, n, m, _ in modes
    ])
    derivative_v = np.column_stack([
        (zernike(n, m, u, v + step) - zernike(n, m, u, v - step)) / (2 * step)
        for _, n, m, _ in modes
    ])
    matrix = np.vstack([derivative_u, derivative_v])
    target = np.r_[-NA_IMAGE * transverse[:, 0],
                   -NA_IMAGE * transverse[:, 1]]
    fit_weights = np.r_[weights, weights]
    coefficients = np.linalg.lstsq(
        matrix * np.sqrt(fit_weights[:, None]),
        target * np.sqrt(fit_weights), rcond=None)[0]
    gradient_residual = target - matrix @ coefficients

    records = []
    for (ansi, n, m, name), coefficient in zip(modes, coefficients):
        records.append({
            "ansi_index": ansi,
            "radial_order_n": n,
            "azimuthal_frequency_m": m,
            "name": name,
            "coefficient_nm_rms": coefficient * 1e6,
            "coefficient_waves_rms": coefficient / WAVELENGTH_MM,
        })
    active = np.array([
        n > 1 or (n == 1 and False) for _, n, _, _ in modes
    ])
    refocused = np.array([
        n > 1 and not (n == 2 and m == 0) for _, n, m, _ in modes
    ])
    return {
        "records": records,
        "modes": modes,
        "coefficients_mm": coefficients,
        "u": u, "v": v, "weights": weights,
        "rms_no_tilt_nm": np.sqrt(np.sum(coefficients[active] ** 2)) * 1e6,
        "rms_refocused_nm": np.sqrt(np.sum(coefficients[refocused] ** 2)) * 1e6,
        "gradient_fit_residual_um": (
            np.sqrt(np.average(gradient_residual ** 2, weights=fit_weights)) * 1e3),
    }


def export_zernike_csv(output: Path, expansion=None):
    expansion = zernike_expansion() if expansion is None else expansion
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(expansion["records"][0]))
        writer.writeheader()
        writer.writerows(expansion["records"])


def wavefront_from_zernikes(u, v, expansion, refocused=False):
    wavefront = np.zeros_like(np.asarray(u, dtype=float))
    for mode, coefficient in zip(expansion["modes"], expansion["coefficients_mm"]):
        _, n, m, _ = mode
        if n == 1 or (refocused and n == 2 and m == 0):
            continue
        wavefront += coefficient * zernike(n, m, u, v)
    return wavefront


def plot_zernikes(output: Path, expansion=None):
    expansion = zernike_expansion() if expansion is None else expansion
    records = expansion["records"]
    coefficients = np.array([row["coefficient_nm_rms"] for row in records])
    labels = [f"Z{row['ansi_index']}\n({row['radial_order_n']},{row['azimuthal_frequency_m']})"
              for row in records]
    grid = np.linspace(-1, 1, 301)
    uu, vv = np.meshgrid(grid, grid)
    disk = uu * uu + vv * vv <= 1
    nominal = wavefront_from_zernikes(uu, vv, expansion, refocused=False) * 1e6
    refocused = wavefront_from_zernikes(uu, vv, expansion, refocused=True) * 1e6
    nominal[~disk] = np.nan
    refocused[~disk] = np.nan

    fig = plt.figure(figsize=(14, 8))
    grid_spec = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    ax0 = fig.add_subplot(grid_spec[0, :])
    ax0.bar(np.arange(len(records)), coefficients, color="#2878b5")
    ax0.set_xticks(np.arange(len(records)), labels, rotation=70, ha="right")
    ax0.set(ylabel="RMS coefficient (nm)", title="ANSI/OSA Zernike coefficients")
    ax0.grid(axis="y", alpha=0.2)
    for ax, data, title in (
        (fig.add_subplot(grid_spec[1, 0]), nominal,
         f"Nominal plane: {expansion['rms_no_tilt_nm']:.2f} nm RMS"),
        (fig.add_subplot(grid_spec[1, 1]), refocused,
         f"After tilt/defocus removal: {expansion['rms_refocused_nm']:.2f} nm RMS"),
    ):
        limit = max(np.nanmax(np.abs(data)), 1e-6)
        shown = ax.imshow(data, extent=[-1, 1, -1, 1], origin="lower",
                          cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set(xlabel="u", ylabel="v", title=title, aspect="equal")
        fig.colorbar(shown, ax=ax, label="Wavefront (nm)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def scalar_imaging(output_psf: Path, output_image: Path, expansion=None):
    """Partially coherent Abbe imaging, matching the companion notebooks."""
    expansion = zernike_expansion() if expansion is None else expansion
    size = 512
    pixel_mm = 10e-6  # 10 nm wafer pixels; 5.12 µm field
    sigma = 0.7
    frequency = np.fft.fftfreq(size, pixel_mm)
    fx, fy = np.meshgrid(frequency, frequency)
    u = fx * WAVELENGTH_MM / NA_IMAGE
    v = fy * WAVELENGTH_MM / NA_IMAGE
    rho2 = u * u + v * v
    wavefront_mm = wavefront_from_zernikes(u, v, expansion, refocused=True)
    pupil = (rho2 <= 1.0) * np.exp(1j * 2 * np.pi * wavefront_mm / WAVELENGTH_MM)

    amplitude = np.fft.fftshift(np.fft.fft2(pupil))
    psf = np.abs(amplitude) ** 2
    psf /= psf.max()
    coordinates_nm = (np.arange(size) - size // 2) * 10.0
    crop = 55
    centre = size // 2
    fig, ax = plt.subplots(figsize=(6, 5))
    extent = [coordinates_nm[centre-crop], coordinates_nm[centre+crop-1],
              coordinates_nm[centre-crop], coordinates_nm[centre+crop-1]]
    shown = ax.imshow(psf[centre-crop:centre+crop, centre-crop:centre+crop],
                      extent=extent, origin="lower", cmap="inferno")
    ax.set(title=("Best-focus scalar PSF from fitted Zernike pupil\n"
                  f"WFE={expansion['rms_refocused_nm']:.2f} nm RMS"),
           xlabel="x at wafer (nm)", ylabel="y at wafer (nm)")
    fig.colorbar(shown, ax=ax, label="Normalized intensity")
    fig.tight_layout()
    fig.savefig(output_psf, dpi=180)
    plt.close(fig)

    # Same style of wafer-scale test target as the companion objective notebook.
    x_nm = (np.arange(size) - size // 2) * 10.0
    xx, yy = np.meshgrid(x_nm, x_nm)
    mask = np.zeros((size, size))

    def rectangle(x0, y0, width, height):
        mask[(np.abs(xx - x0) <= width / 2)
             & (np.abs(yy - y0) <= height / 2)] = 1.0

    for k in range(-5, 6):
        rectangle(-1450 + k * 140, 1300, 70, 1150)   # vertical 70 nm L/S
        rectangle(1450, 1300 + k * 140, 1150, 70)    # horizontal 70 nm L/S
    for k in range(5):
        rectangle(-1950 + k * 210, -1400, 100, 1050)
        rectangle(-1530, -1950 + k * 210, 950, 100)  # elbows
    rectangle(250, -1350, 70, 1200)                  # isolated line
    for kx in range(5):
        for ky in range(5):
            rectangle(1100 + kx * 260, -1950 + ky * 260, 120, 120)

    mask_spectrum = np.fft.fft2(mask)
    aerial = np.zeros_like(mask)
    df = frequency[1] - frequency[0]
    source_count = 0
    for sx in np.linspace(-sigma, sigma, 11):
        for sy in np.linspace(-sigma, sigma, 11):
            if sx * sx + sy * sy > sigma * sigma:
                continue
            shift_x = int(round(sx * NA_IMAGE / WAVELENGTH_MM / df))
            shift_y = int(round(sy * NA_IMAGE / WAVELENGTH_MM / df))
            shifted_pupil = np.roll(np.roll(pupil, shift_y, axis=0),
                                     shift_x, axis=1)
            aerial += np.abs(np.fft.ifft2(mask_spectrum * shifted_pupil)) ** 2
            source_count += 1
    aerial /= aerial.max()
    threshold = 0.30
    extent_nm = [x_nm[0], x_nm[-1], x_nm[0], x_nm[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(mask, extent=extent_nm, origin="lower", cmap="gray_r")
    axes[0].set_title("Mask (wafer scale)")
    shown = axes[1].imshow(aerial, extent=extent_nm, origin="lower", cmap="inferno")
    axes[1].set_title(
        f"Abbe aerial image (σ={sigma}, {source_count} source points)")
    fig.colorbar(shown, ax=axes[1], fraction=0.046)
    axes[2].imshow(aerial > threshold, extent=extent_nm,
                   origin="lower", cmap="gray_r")
    axes[2].set_title(f"Developed resist (threshold {threshold:.2f})")
    for ax in axes:
        ax.set_xlabel("x (nm)")
    axes[0].set_ylabel("y (nm)")
    fig.tight_layout()
    fig.savefig(output_image, dpi=180)
    plt.close(fig)


def write_report(output: Path):
    rays = pupil_rays(62.0, 10, 48)
    points = np.array([ray[2]["image"][:2] for ray in rays])
    centre = np.median(points, axis=0)
    airy_nm = 0.61 * WAVELENGTH_MM / NA_IMAGE * 1e6
    metrics = aberration_metrics()
    expansion = zernike_expansion()
    rms_um = metrics[1]["rms_spot_radius_um"]
    metric_lines = "\n".join(
        f"| {row['centroid_image_height_mm']:.6f} | "
        f"{row['rms_spot_radius_um']:.4f} | {row['best_focus_rms_spot_um']:.4f} | "
        f"{row['ee80_radius_um']:.4f} | "
        f"{row['distortion_um']:+.6f} | {row['sagittal_focus_shift_mm']:+.6e} | "
        f"{row['tangential_focus_shift_mm']:+.6e} | "
        f"{row['astigmatic_separation_mm']:+.6e} |"
        for row in metrics
    )
    output.write_text(
        "# US7557996 objective model\n\n"
        "Canonical embodiment: Figure 3 / Tables 3 and 3A.\n\n"
        f"- Wavelength: 193.368 nm\n"
        f"- Image-side NA: {NA_IMAGE}\n"
        f"- Nominal reduction: {REDUCTION:.0f}×\n"
        f"- Sequential surfaces: {len(SURFACES)} (including 2 mirrors)\n"
        f"- Unique physical lenses: {len(physical_lenses())}\n"
        f"- Recovered object plane z: {OBJECT_Z:.6f} mm\n"
        f"- Numerical paraxial magnification: {PARAXIAL_MAGNIFICATION:.8f}\n"
        f"- Image plane z: {IMAGE_Z:.6f} mm\n"
        f"- Centre-field retained pupil rays: {len(rays)}\n"
        f"- Centre-field geometric RMS radius: {rms_um:.4f} µm\n"
        f"- Ideal scalar Airy radius: {airy_nm:.2f} nm\n\n"
        "## Monochromatic aberrations\n\n"
        "| Image height (mm) | RMS spot (µm) | Best-focus RMS (µm) | EE80 radius (µm) | "
        "Distortion (µm) | Sagittal focus Δz (mm) | "
        "Tangential focus Δz (mm) | T−S astigmatism (mm) |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        f"{metric_lines}\n\n"
        "## Zernike wavefront reconstruction\n\n"
        f"- ANSI/OSA expansion through radial order 6\n"
        f"- RMS excluding tilt: {expansion['rms_no_tilt_nm']:.4f} nm\n"
        f"- RMS after tilt and defocus removal: "
        f"{expansion['rms_refocused_nm']:.4f} nm\n"
        f"- Transverse-gradient fit residual: "
        f"{expansion['gradient_fit_residual_um']:.6f} µm\n\n"
        "The prescription is copied verbatim from the patent. The object plane is "
        "not listed in Table 3 and is recovered from the paraxial conjugate "
        "condition. The aerial image uses scalar partially coherent Abbe imaging "
        "with conventional illumination (sigma=0.7), the fitted best-focus "
        "Zernike pupil, and a synthetic binary mask. Polarization, mask topography, "
        "and vector high-NA effects are outside the model. Chromatic aberrations cannot be derived "
        "because the prescription gives refractive indices at only one wavelength.\n",
        encoding="utf-8",
    )


def main():
    output = Path("outputs_us7557996")
    output.mkdir(exist_ok=True)
    export_prescription_csv(Path("US7557996_Fig3_Table3_prescription.csv"))
    metrics = aberration_metrics()
    expansion = zernike_expansion()
    export_aberration_csv(Path("US7557996_aberration_metrics.csv"), metrics)
    export_zernike_csv(Path("US7557996_zernike_coefficients.csv"), expansion)
    plot_layout(output / "layout_raytrace.png")
    plot_spots(output / "spot_diagrams.png")
    plot_aberrations(output / "aberration_summary.png",
                     output / "ray_aberration_fans.png", metrics)
    plot_zernikes(output / "zernike_wavefront.png", expansion)
    scalar_imaging(output / "scalar_psf.png", output / "aerial_image.png",
                   expansion)
    write_report(output / "REPORT.md")
    print(f"Wrote results to {output.resolve()}")
    print(f"Object z={OBJECT_Z:.6f} mm, paraxial M={PARAXIAL_MAGNIFICATION:.8f}")


if __name__ == "__main__":
    main()
