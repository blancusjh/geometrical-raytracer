"""Matplotlib geometry plots using the tracer's actual three-dimensional frames."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from ..surfaces.apertures import CircularAperture, RectangularAperture


def _surface_curves(row, samples):
    aperture = row.clear_aperture
    if isinstance(aperture, CircularAperture):
        radius = aperture.radius
        phi = np.linspace(0, 2 * np.pi, samples)
        curves = [np.column_stack([radius * np.cos(phi), radius * np.sin(phi)])]
        if aperture.inner_radius:
            inner = aperture.inner_radius
            curves.append(np.column_stack([inner * np.cos(phi), inner * np.sin(phi)]))
    elif isinstance(aperture, RectangularAperture):
        radius = np.hypot(aperture.half_width, aperture.half_height)
        x, y = aperture.half_width, aperture.half_height
        corners = np.array([[-x, -y], [x, -y], [x, y], [-x, y], [-x, -y]])
        curves = [
            np.vstack(
                [np.linspace(a, b, max(samples // 4, 2)) for a, b in zip(corners[:-1], corners[1:])]
            )
        ]
    else:
        raise ValueError("3-D layouts need an explicit circular or rectangular display aperture")
    heights = np.linspace(-radius, radius, samples)
    for angle in (0, np.pi / 2):
        xy = np.column_stack([heights * np.cos(angle), heights * np.sin(angle)])
        xy[~aperture.contains(xy, slack=1e-12)] = np.nan
        curves.append(xy)
    return curves


def layout_3d_figure(system, *, paths=None, samples=120, title=None):
    """World-coordinate wireframes: overall ray layout and magnified surface view.

    Display axes are ordered (z, x, y) to put the nominal optical axis across
    the page; the coordinates themselves remain global. Undefined sag samples
    remain gaps. Mechanical solids, mounts and clearances are not inferred.
    The detector outline is a display patch, not an aperture in the ray model.
    """
    if samples < 8:
        raise ValueError("at least eight surface samples are required")
    system._rebuild()
    fig = plt.figure(figsize=(13, 5.5), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.3, 1))
    axes = [fig.add_subplot(grid[0, index], projection="3d") for index in range(2)]
    surface_points = []
    for index, row in enumerate(system.rows):
        frame = system.surface_frame(index)
        color = "#777777" if row.reflective else "#257a98"
        for xy in _surface_curves(row, samples):
            local = np.column_stack([xy, row.profile.sag(np.linalg.norm(xy, axis=1))])
            world = frame.to_world(local)[:, [2, 0, 1]]
            for ax in axes:
                ax.plot(*world.T, color=color, linewidth=1)
            surface_points.extend(world[np.all(np.isfinite(world), axis=1)])
        axes[1].text(*frame.origin[[2, 0, 1]], str(index + 1), fontsize=8)
    surface_points = np.asarray(surface_points)
    all_points = [surface_points]
    if paths is not None:
        for path in np.asarray(paths):
            coordinates = path[:, [2, 0, 1]]
            axes[0].plot(*coordinates.T, color="#cf752f", alpha=0.45, linewidth=0.7)
            # Restrict the detail to inter-surface segments; mplot3d does not
            # clip long launch/detector segments to the displayed axis bounds.
            axes[1].plot(*coordinates[1:-1].T, color="#cf752f", alpha=0.45, linewidth=0.7)
            all_points.append(coordinates[np.all(np.isfinite(coordinates), axis=1)])
    detector = system.image_frame
    half_width = max(np.ptp(surface_points[:, 1:], axis=0).max() / 2, 1)
    corners = np.array([[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]]) * half_width
    detector_patch = detector.to_world(corners)[:, [2, 0, 1]]
    axes[0].plot(*detector_patch.T, color="#666666", linestyle="--", label="Plano detector")
    all_points.append(detector_patch)
    for ax, points, label in zip(
        axes,
        [np.vstack(all_points), surface_points],
        ["Trayectoria completa", "Detalle de superficies"],
    ):
        lower, upper = points.min(axis=0), points.max(axis=0)
        extent = upper - lower
        padding = np.maximum(extent * 0.08, 0.05)
        for setter, lo, hi, margin in zip(
            [ax.set_xlim, ax.set_ylim, ax.set_zlim], lower, upper, padding
        ):
            setter(lo - margin, hi + margin)
        ax.set_box_aspect(np.maximum(extent, max(extent.max(), 1) * 0.08))
        ax.view_init(elev=22, azim=-68)
        ax.set(xlabel="z [mm]", ylabel="x [mm]", zlabel="y [mm]", title=label)
        ax.tick_params(labelsize=8)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(3))
            axis.labelpad = 14
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(title or system.name or "Geometrical layout")
    return fig


__all__ = ["layout_3d_figure"]
