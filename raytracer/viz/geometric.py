"""Matplotlib geometry plots using the tracer's actual three-dimensional frames."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..surfaces.apertures import CircularAperture, RectangularAperture
from .solid_geometry import glass_blue, lens_meshes


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


def _clip_segment(start, end, lower, upper):
    """Clip a display segment to the detail box without moving the traced ray."""
    if not np.all(np.isfinite([start, end])):
        return None
    direction = end - start
    entry, exit = 0.0, 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1e-15:
            if not lower[axis] <= start[axis] <= upper[axis]:
                return None
            continue
        a, b = sorted(
            [
                (lower[axis] - start[axis]) / direction[axis],
                (upper[axis] - start[axis]) / direction[axis],
            ]
        )
        entry, exit = max(entry, a), min(exit, b)
        if entry > exit:
            return None
    return np.array([start + entry * direction, start + exit * direction])


def layout_3d_figure(system, *, paths=None, samples=120, title=None, ray_colors=None):
    """Matplotlib lens volumes, complete paths and intersections on the detector.

    Display axes are (world z, world x, world y), with equal geometric scale.
    Mesh vertices and outlines use the same sag and placements as the tracer.
    The full view retains launch and image segments. Only the lens close-up
    clips segments to its viewing box. The detector patch is a display extent,
    not a physical aperture. Alpha blending is illustrative, not radiometry.
    """
    if samples < 8:
        raise ValueError("at least eight surface samples are required")
    system._rebuild()
    paths = np.empty((0, 0, 3)) if paths is None else np.asarray(paths, dtype=float)
    if paths.ndim != 3 or paths.shape[-1] != 3:
        raise ValueError("paths must have shape (rays, vertices, 3)")
    colors = ["#b96c24"] * len(paths) if ray_colors is None else list(ray_colors)
    if len(colors) != len(paths):
        raise ValueError("supply one ray color per path")
    fig = plt.figure(figsize=(13, 7.5))
    fig.subplots_adjust(left=0.07, right=0.96, top=0.90, bottom=0.09, hspace=0.27, wspace=0.26)
    grid = fig.add_gridspec(2, 2, height_ratios=(0.8, 1))
    overall = fig.add_subplot(grid[0, :], projection="3d")
    detail = fig.add_subplot(grid[1, 0], projection="3d")
    detector_ax = fig.add_subplot(grid[1, 1])
    axes = [overall, detail]
    surface_points = []
    # Keep the established contour renderer for every supported aperture.
    for index, row in enumerate(system.rows):
        frame = system.surface_frame(index)
        for xy in _surface_curves(row, samples):
            local = np.column_stack([xy, row.profile.sag(np.linalg.norm(xy, axis=1))])
            world = frame.to_world(local)[:, [2, 0, 1]]
            for ax in axes:
                ax.plot(*world.T, color="#365c79", linewidth=0.7, alpha=0.85)
            surface_points.extend(world[np.all(np.isfinite(world), axis=1)])
    surface_points = np.asarray(surface_points)
    # One collection sorts triangles from all volumes together in projection.
    try:
        meshes = lens_meshes(system, radial_samples=12, angular_samples=max(samples // 2, 32))
    except ValueError as error:
        warnings.warn(f"Lens fill omitted; exact contours retained: {error}", stacklevel=2)
        meshes = []
    if meshes:
        triangles = np.concatenate([m.vertices[m.triangles][:, :, [2, 0, 1]] for m in meshes])
        facecolors = np.concatenate(
            [
                np.tile((*glass_blue(m.refractive_index), 0.13), (len(m.triangles), 1))
                for m in meshes
            ]
        )
        for ax in axes:
            ax.add_collection3d(
                Poly3DCollection(
                    triangles,
                    facecolors=facecolors,
                    edgecolors="none",
                    linewidths=0,
                    antialiased=True,
                )
            )
        legend = {(m.material, m.refractive_index) for m in meshes}
        detail.legend(
            handles=[
                Patch(facecolor=glass_blue(n), alpha=0.6, label=f"{name}: n = {n:.5f}")
                for name, n in sorted(legend)
            ],
            loc="upper left",
            fontsize=8,
            frameon=False,
        )
    lower, upper = surface_points.min(axis=0), surface_points.max(axis=0)
    padding = np.maximum((upper - lower) * 0.20, 0.6)
    detail_lower, detail_upper = lower - padding, upper + padding
    all_points = [surface_points]
    hits = []
    hit_colors = []
    for path, color in zip(paths, colors):
        coordinates = path[:, [2, 0, 1]]
        overall.plot(*coordinates.T, color=color, linewidth=0.8, alpha=0.85)
        all_points.append(coordinates[np.all(np.isfinite(coordinates), axis=1)])
        for start, end in zip(coordinates[:-1], coordinates[1:]):
            clipped = _clip_segment(start, end, detail_lower, detail_upper)
            if clipped is not None:
                detail.plot(*clipped.T, color=color, linewidth=0.85, alpha=0.85)
        if len(path) and np.all(np.isfinite(path[-1])):
            hits.append(path[-1])
            hit_colors.append(color)
    hits = np.asarray(hits).reshape(-1, 3)
    detector = system.image_frame
    local_hits = detector.to_local(hits)[:, :2]
    # Enclose every actual image-plane intersection, including off-axis fields.
    radius = max(np.ptp(surface_points[:, 1:], axis=0).max() / 2, 1)
    xy = np.vstack([[-radius, -radius], [radius, radius], local_hits])
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    margin = (hi - lo) * 0.08
    lo, hi = lo - margin, hi + margin
    corners = np.array(
        [
            [lo[0], lo[1], 0],
            [hi[0], lo[1], 0],
            [hi[0], hi[1], 0],
            [lo[0], hi[1], 0],
            [lo[0], lo[1], 0],
        ]
    )
    patch = detector.to_world(corners)[:, [2, 0, 1]]
    overall.plot(*patch.T, color="#657a87", linewidth=1, label="Pantalla")
    if len(hits):
        overall.scatter(*hits[:, [2, 0, 1]].T, c=hit_colors, s=8, depthshade=False)
        detector_ax.scatter(*local_hits.T, c=hit_colors, s=12, alpha=0.8)
    all_points.append(patch)
    for ax, points, label in [
        (overall, np.vstack(all_points), "Rayos incidentes → lentes → pantalla"),
        (detail, np.array([detail_lower, detail_upper]), "Detalle de las lentes · misma geometría"),
    ]:
        low, high = points.min(axis=0), points.max(axis=0)
        extent = high - low
        margin = np.maximum(extent * 0.04, 0.05)
        for setter, a, b, pad in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], low, high, margin):
            setter(a - pad, b + pad)
        ax.set_box_aspect(np.maximum(extent, 0.05), zoom=1.0 if ax is overall else 1.1)
        ax.set_proj_type("ortho")
        ax.view_init(elev=15, azim=-74)
        ax.set(xlabel="z [mm]", ylabel="x [mm]", zlabel="y [mm]", title=label)
        ax.tick_params(labelsize=8, pad=1)
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(3))
            axis.labelpad = 5
            axis.pane.fill = False
    for artist in [*overall.lines, *overall.collections]:
        artist.set_clip_on(False)
    overall.set_axis_off()
    overall.legend(loc="upper left", frameon=False, fontsize=9)
    overall.text2D(
        0.5,
        0.10,
        "Vista ortográfica · proporciones geométricas conservadas",
        ha="center",
        fontsize=9,
        color="#536471",
        transform=overall.transAxes,
    )
    detector_ax.set(
        xlabel="x local de la pantalla [mm]",
        ylabel="y local de la pantalla [mm]",
        title="Intersecciones reales en la pantalla",
    )
    detector_ax.set_aspect("equal", adjustable="datalim")
    detector_ax.grid(alpha=0.15)
    if not len(hits):
        detector_ax.text(
            0.5,
            0.5,
            "Sin intersecciones suministradas",
            ha="center",
            transform=detector_ax.transAxes,
        )
    fig.suptitle(title or system.name or "Geometrical layout")
    # mplot3d uses a square axes box even in a wide subplot. Fit the projected
    # complete geometry to the available rectangular row, including its height.
    fig.canvas.draw()
    projected = proj3d.proj_transform(*np.vstack(all_points).T, overall.get_proj())
    pixels = overall.transData.transform(np.column_stack(projected[:2]))
    available = overall.get_subplotspec().get_position(fig).transformed(fig.transFigure)
    span = np.maximum(np.ptp(pixels, axis=0), 1)
    zoom = min(available.width * 0.85 / span[0], available.height * 0.60 / span[1])
    overall.set_box_aspect(np.maximum(np.ptp(np.vstack(all_points), axis=0), 0.05), zoom=zoom)
    return fig


__all__ = ["layout_3d_figure"]
