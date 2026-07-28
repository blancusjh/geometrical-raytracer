"""Matplotlib analysis figures for sequential systems.

Every function returns a :class:`matplotlib.figure.Figure` so callers (and
notebooks) decide whether to show, save, or embed it.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..propagation.fields import FieldPoint, chief_ray_slope, trace_from_object
from ..design.rows import SurfaceKind
from ..design.system import OpticalSystem
from ..propagation.sequential import SequentialTracer
from ..analysis.aberrations.chromatic import AxialColor, ChromaticSpots, LateralColor
from ..analysis.aberrations.distortion import DistortionGrid
from ..analysis.aberrations.fans import FanData
from ..analysis.aberrations.wavefront import WavefrontSamples
from ..analysis.aberrations.zernike import ZernikeExpansion
from ..analysis.imaging.spots import SpotData
from .bodies import dense_gaps, element_outline, index_color
from .sag_drawing import sample_profile_curve

DEFAULT_MATERIAL_COLORS = {
    "SIO2": ("#bde0fe", "#2878b5"),
    "CAF2": ("#c7f9cc", "#278b65"),
    "HIINDEX1": ("#ffe29a", "#c47f00"),
    "HIINDEX2": ("#ffb3a1", "#c1440e"),
}
FIELD_COLORS = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]


def layout_figure(
    tracer: SequentialTracer,
    *,
    fields: Sequence[float] = (),
    fan_half_slope: float = 0.29,
    fan_count: int = 7,
    material_colors: dict | None = None,
    mirror_labels: bool = True,
    figsize: tuple[float, float] = (14, 7),
    title: str | None = None,
) -> plt.Figure:
    """Closed lens layout with filled bodies, mirrors, and meridional rays.

    Bodies are deduced from the media sequence (every inter-surface gap
    with index above 1) and colored by optical density via
    :func:`raytracer.viz.bodies.index_color` — denser media draw darker,
    automatically. ``material_colors`` overrides individual media by name
    (``DEFAULT_MATERIAL_COLORS`` keeps the README's DUV palette).
    """

    system = tracer.system
    colors = dict(DEFAULT_MATERIAL_COLORS)
    if material_colors:
        colors.update(material_colors)
    fig, ax = plt.subplots(figsize=figsize)

    solid_signatures: set[tuple[float, float]] = set()
    for i, j, n, material in dense_gaps(system):
        for k in (i, j):
            solid_signatures.add(
                (round(float(system.vertices[k]), 6), round(system.rows[k].radius, 6))
            )
        outline, _, _ = element_outline(system, i, j)
        face, edge = colors.get(material.upper()) or index_color(n)
        ax.fill(
            outline[:, 0],
            outline[:, 1],
            facecolor=face,
            edgecolor=edge,
            lw=1.0,
            alpha=0.66,
            zorder=1,
        )

    # Bare refracting surfaces not part of a solid_elements() pair (e.g. a
    # single dioptric surface with no matching second face) still need to be
    # drawn, or they'd be invisible — plotted as an open curve, like a mirror.
    # Skipped: index-matched dummy rows (no optical effect), and double-passed
    # copies of surfaces already drawn as a solid element on a folded path
    # (matched by the same geometric signature solid_elements() dedups with).
    for i, row in enumerate(system.rows):
        if row.kind is not SurfaceKind.REFRACT:
            continue
        if system.n_before[i] == system.n_after[i]:
            continue
        signature = (round(float(system.vertices[i]), 6), round(row.radius, 6))
        if signature in solid_signatures:
            continue
        semi = row.semidiameter if row.semidiameter is not None else 50.0
        z, h = sample_profile_curve(system, i, semidiameter=semi)
        ax.plot(z, h, color="#3a6ea5", lw=2.0, zorder=2)

    for count, i in enumerate(system.mirror_indices, 1):
        row = system.rows[i]
        semi = row.semidiameter if row.semidiameter is not None else 50.0
        z, h = sample_profile_curve(system, i, semidiameter=semi)
        ax.plot(z, h, color="#202020", lw=3.0, zorder=2)
        if mirror_labels:
            ax.text(float(np.mean(z)), semi + 8, f"M{count}", ha="center", fontsize=9)

    # Immersion / non-air image space.
    if system.n_after[-1] != 1.0:
        last = system.rows[-1]
        semi = last.semidiameter if last.semidiameter is not None else 20.0
        ax.fill_betweenx(
            [-semi, semi], system.vertices[-1], system.image_z,
            color="#f6bd60", alpha=0.24, zorder=0,
        )

    if fields:
        for field_y, color in zip(fields, FIELD_COLORS):
            sy0 = chief_ray_slope(tracer, FieldPoint(y=field_y))
            for delta in np.linspace(-fan_half_slope, fan_half_slope, fan_count):
                result = trace_from_object(
                    tracer, (0.0, field_y), (0.0, sy0 + delta), keep_path=True
                )
                if result.path is None:
                    continue
                path = result.path[~np.isnan(result.path[:, 2])]
                ax.plot(path[:, 2], path[:, 1], color=color, lw=0.65, alpha=0.72, zorder=3)

    if system.object_z is not None:
        ax.axvline(system.object_z, color="black", ls="--", lw=1, label="object plane")
    ax.axvline(system.image_z, color="black", ls="--", lw=1, label="image plane")
    if system.stop_index is not None:
        ax.axvline(
            system.vertices[system.stop_index], color="#555555", ls=":", lw=1,
            label="aperture stop",
        )
    ax.axhline(0, color="#aaaaaa", lw=0.6)
    ax.set(
        xlabel="Global z (mm)",
        ylabel="Meridional y (mm)",
        title=title or (system.name or "Sequential layout"),
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


def mirror_arcs_from_paths(
    system: OpticalSystem,
    paths,
    *,
    pad: float = 0.06,
    samples: int = 200,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Per-mirror meridional arcs spanning the heights the rays actually hit.

    For off-axis systems (ring fields, folded paths) a mirror's *used*
    sub-aperture is displaced from the axis, so drawing the parent surface
    as a symmetric on-axis cap puts the drawn curve away from the real
    reflection points. This derives each mirror's arc from the traced
    *paths* themselves (``keep_paths`` output, shape ``(N, n_surfaces+2, 3)``;
    column ``i+1`` is the hit point on surface ``i``), extended by *pad*
    fractionally beyond the hit envelope.

    Returns ``[(mirror_number, z, y), ...]`` ready for ``ax.plot(z, y)``.
    """

    paths = np.asarray(paths, dtype=float)
    arcs: list[tuple[int, np.ndarray, np.ndarray]] = []
    for count, i in enumerate(system.mirror_indices, 1):
        y_hits = paths[:, i + 1, 1]
        y_hits = y_hits[np.isfinite(y_hits)]
        if y_hits.size == 0:
            continue
        y_lo, y_hi = float(y_hits.min()), float(y_hits.max())
        margin = pad * max(y_hi - y_lo, 1.0)
        h = np.linspace(y_lo - margin, y_hi + margin, samples)
        z = system.vertices[i] + system.rows[i].profile.sag(np.abs(h))
        arcs.append((count, z, h))
    return arcs


def spots_figure(
    spots: Iterable[SpotData],
    *,
    airy_radius_um: float | None = None,
    figsize_per_panel: float = 4.0,
    colors: Sequence | None = None,
) -> plt.Figure:
    """Spot diagrams recentred on the weighted centroid, one panel per field.

    Pass ``colors=FIELD_COLORS`` (one entry per field, same order as *spots*)
    so each panel keeps the color its field's rays carry in
    :func:`layout_figure` — the default leaves matplotlib's single color.
    """

    spots = list(spots)
    if colors is None:
        colors = [None] * len(spots)
    fig, axes = plt.subplots(
        1, len(spots), figsize=(figsize_per_panel * len(spots), figsize_per_panel)
    )
    if len(spots) == 1:
        axes = [axes]
    for spot, ax, color in zip(spots, axes, colors):
        ax.scatter(spot.relative_um[:, 0], spot.relative_um[:, 1], s=4, alpha=0.55, color=color)
        if airy_radius_um is not None:
            circle = plt.Circle(
                (0, 0), airy_radius_um, fill=False, color="#d62728", ls="--", lw=1.0
            )
            ax.add_patch(circle)
        ax.set_title(
            f"Object y={spot.field_y:.0f} mm\narea-weighted RMS={spot.rms_radius_um:.3f} µm"
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_xlabel("Δx (µm)")
    axes[0].set_ylabel("Δy (µm)")
    fig.suptitle("Geometric spot diagrams — unvignetted rays")
    fig.tight_layout()
    return fig


def fans_figure(fans: Iterable[FanData]) -> plt.Figure:
    """Tangential and sagittal transverse-aberration fans."""

    fans = list(fans)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for fan, color in zip(fans, FIELD_COLORS):
        axes[0].plot(
            fan.tangential[:, 0], fan.tangential[:, 1], color=color,
            label=f"y={fan.field_y:.0f} mm",
        )
        axes[1].plot(
            fan.sagittal[:, 0], fan.sagittal[:, 1], color=color,
            label=f"y={fan.field_y:.0f} mm",
        )
    axes[0].set(title="Tangential ray fan", ylabel="Δy′ (µm)")
    axes[1].set(title="Sagittal ray fan", ylabel="Δx′ (µm)")
    for ax in axes:
        ax.set_xlabel("Normalized pupil coordinate")
        ax.axhline(0, color="black", lw=0.7)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    return fig


def aberrations_figure(metrics: Sequence[dict], *, suptitle: str | None = None) -> plt.Figure:
    """2x2 field summary: RMS spot, distortion, field curvature, astigmatism."""

    image_height = np.array([row["centroid_image_height_mm"] for row in metrics])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(
        image_height, [row["rms_spot_radius_um"] for row in metrics], "o-",
        label="nominal plane",
    )
    axes[0, 0].plot(
        image_height, [row["best_focus_rms_spot_um"] for row in metrics], "s-",
        label="best focus",
    )
    axes[0, 0].set(ylabel="RMS radius (µm)", title="Transverse aberration")
    axes[0, 0].legend()
    axes[0, 1].plot(image_height, [row["chief_ray_distortion_um"] for row in metrics], "o-")
    axes[0, 1].axhline(0, color="black", lw=0.7)
    axes[0, 1].set(ylabel="Chief-ray distortion (µm)", title="Distortion")
    axes[1, 0].plot(
        image_height, [row["sagittal_focus_shift_mm"] for row in metrics], "o-",
        label="sagittal",
    )
    axes[1, 0].plot(
        image_height, [row["tangential_focus_shift_mm"] for row in metrics], "s-",
        label="tangential",
    )
    axes[1, 0].axhline(0, color="black", lw=0.7)
    axes[1, 0].set(ylabel="Best-focus shift (mm)", title="Field curvature")
    axes[1, 0].legend()
    axes[1, 1].plot(
        image_height, [row["astigmatic_separation_mm"] for row in metrics], "o-"
    )
    axes[1, 1].axhline(0, color="black", lw=0.7)
    axes[1, 1].set(ylabel="T−S focus (mm)", title="Astigmatism")
    for ax in axes.flat:
        ax.set_xlabel("Image height (mm)")
        ax.grid(alpha=0.2)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def _draw_distortion_grid(
    ax, grid: DistortionGrid, *, exaggeration: float, color: str = "#1f77b4"
) -> None:
    """Draw one grid on *ax*: ideal dashed underneath, traced solid over it."""

    ideal = grid.ideal_points
    n = ideal.shape[0]
    # Vignetted points are already NaN in actual_points, and NaN survives the
    # scaling, so the broken lines fall out for free.
    displayed = ideal + exaggeration * (grid.actual_points - ideal)

    for i in range(n):
        ax.plot(ideal[i, :, 0], ideal[i, :, 1], color="#999999", ls="--", lw=0.7, zorder=1)
        ax.plot(displayed[i, :, 0], displayed[i, :, 1], color=color, lw=1.4, zorder=2)
    for j in range(n):
        ax.plot(ideal[:, j, 0], ideal[:, j, 1], color="#999999", ls="--", lw=0.7, zorder=1)
        ax.plot(displayed[:, j, 0], displayed[:, j, 1], color=color, lw=1.4, zorder=2)

    if not grid.valid.all():
        dropped = ideal[~grid.valid]
        n_dropped = int((~grid.valid).sum())
        ax.scatter(
            dropped[:, 0], dropped[:, 1], marker="x", color="#d62728", s=24,
            zorder=3, label=f"vignetted ({n_dropped}/{grid.valid.size})",
        )
        ax.legend(loc="best", fontsize=8)

    ax.set_aspect("equal")
    ax.set_xlabel("image x (mm)")
    ax.grid(alpha=0.15)


def distortion_grid_figure(
    grid: DistortionGrid,
    *,
    exaggeration: float = 1.0,
    title: str | None = None,
    figsize: tuple[float, float] = (6.5, 6.5),
) -> plt.Figure:
    """Square-grid distortion display: trace a grid, see it warp.

    The ideal (undistorted) grid is drawn as dashed gray lines; the traced
    chief-ray grid is drawn as solid lines over it. ``exaggeration``
    amplifies each point's departure from ideal (``ideal + exaggeration *
    (actual - ideal)``) around its own ideal position — real distortion is
    usually far too small to see at 1:1 scale, and optical design software
    conventionally offers this as an adjustable dial, not a fixed
    multiplier. Vignetted grid points (see ``grid.valid``) break the line
    they'd otherwise be part of rather than being interpolated over, and
    are marked with an ``x`` so missing coverage is visible, not silent.
    """

    fig, ax = plt.subplots(figsize=figsize)
    _draw_distortion_grid(ax, grid, exaggeration=exaggeration)
    ax.set_ylabel("image y (mm)")
    subtitle = f" (distortion ×{exaggeration:g})" if exaggeration != 1.0 else ""
    ax.set_title((title or "Distortion grid") + subtitle)
    fig.tight_layout()
    return fig


def distortion_grids_figure(
    grids: Iterable[DistortionGrid],
    *,
    titles: Sequence[str] | None = None,
    exaggerations: Sequence[float] | float = 1.0,
    figsize_per_panel: float = 4.4,
    suptitle: str | None = None,
) -> plt.Figure:
    """Several distortion grids side by side, one panel each.

    ``exaggerations`` is a single dial applied to every panel, or one per
    grid: systems worth comparing span orders of magnitude in distortion,
    and a shared multiplier that makes one panel readable flattens or
    explodes the others. Each panel title carries the true, unexaggerated
    magnitude — the largest departure from the ideal map, in µm and as a
    fraction of image height — so the amplified shape is read next to the
    number it was amplified from.
    """

    grids = list(grids)
    if titles is None:
        titles = [""] * len(grids)
    if isinstance(exaggerations, (int, float)):
        exaggerations = [float(exaggerations)] * len(grids)

    fig, axes = plt.subplots(
        1, len(grids), figsize=(figsize_per_panel * len(grids), figsize_per_panel + 0.6)
    )
    if len(grids) == 1:
        axes = [axes]
    for grid, ax, title, exaggeration in zip(grids, axes, titles, exaggerations):
        _draw_distortion_grid(ax, grid, exaggeration=exaggeration)
        ax.set_title(
            f"{title}\nmax {grid.max_distortion_um:.3g} µm "
            f"= {grid.max_relative_distortion_percent:.3g} % of height"
            f"   —   drawn ×{exaggeration:g}",
            fontsize=10,
        )
    axes[0].set_ylabel("image y (mm)")
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def _mark_reference_lines(ax, wavelengths_um: Sequence[float]) -> None:
    for wavelength_um in wavelengths_um:
        ax.axvline(wavelength_um, color="#999999", ls=":", lw=0.8)


def dispersion_figure(
    materials: Sequence,
    wavelengths_um: np.ndarray,
    *,
    reference_lines_um: Sequence[float] = (),
    title: str | None = None,
) -> plt.Figure:
    """Refractive index vs. wavelength for each material.

    Each material only needs an ``index(wavelength_um)`` method, so a
    measured :class:`~raytracer.optics.materials.SellmeierMaterial` and an
    approximate :class:`~raytracer.optics.materials.AbbeMaterial` plot the
    same way; the legend names the model actually behind each curve, since
    the difference matters when reading the chromatic results downstream.
    """

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for material in materials:
        index = [material.index(float(w)) for w in wavelengths_um]
        ax.plot(wavelengths_um, index, "-", lw=1.6,
                label=f"{material.name} ({type(material).__name__})")
    _mark_reference_lines(ax, reference_lines_um)
    ax.set(xlabel="wavelength (µm)", ylabel="refractive index n",
           title=title or "Glass dispersion")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def chromatic_figure(
    axial: AxialColor,
    lateral: LateralColor | None = None,
    *,
    reference_lines_um: Sequence[float] = (),
    suptitle: str | None = None,
) -> plt.Figure:
    """Axial color, and optionally lateral color, against wavelength."""

    panels = 1 if lateral is None else 2
    fig, axes = plt.subplots(1, panels, figsize=(5.5 * panels, 4.2), squeeze=False)
    flat = axes[0]

    flat[0].plot(axial.wavelengths_um, axial.focus_shift_mm * 1e3, "-",
                 color="#b3421b", lw=1.6)
    flat[0].set(xlabel="wavelength (µm)", ylabel="focus shift (µm)", title="Axial color")

    if lateral is not None:
        flat[1].plot(lateral.wavelengths_um, lateral.lateral_color_um, "-",
                     color="#2878b5", lw=1.6)
        flat[1].set(xlabel="wavelength (µm)", ylabel="image height shift (µm)",
                    title=f"Lateral color at y={lateral.field_height_mm:.4g} mm")

    for ax in flat:
        ax.axhline(0, color="black", lw=0.7)
        _mark_reference_lines(ax, reference_lines_um)
        ax.grid(alpha=0.2)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def chromatic_spots_figure(
    spots: ChromaticSpots,
    *,
    limit_um: float | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (6.0, 6.0),
) -> plt.Figure:
    """Every wavelength's spot overlaid in one common reference frame.

    Colors run blue-to-red across the sampled spectrum, so the plot reads
    the way the aberration does. Pass ``limit_um`` to hold the axes fixed
    across several figures — otherwise each autoscales and a tight spot and
    a hugely blurred one look deceptively alike.
    """

    wavelengths = spots.wavelengths_um
    lo, hi = float(wavelengths.min()), float(wavelengths.max())
    colormap = plt.get_cmap("turbo")

    fig, ax = plt.subplots(figsize=figsize)
    for wavelength_um, offsets in zip(wavelengths, spots.offsets_um):
        fraction = 0.5 if hi == lo else (wavelength_um - lo) / (hi - lo)
        # turbo runs blue -> green -> red, matching the spectrum's own order,
        # so the shortest wavelength must take the *low* end. Both ends are
        # squeezed inward off turbo's near-black extremes to stay legible.
        ax.scatter(
            offsets[:, 0], offsets[:, 1], s=5, alpha=0.6,
            color=colormap(0.06 + 0.88 * fraction),
            label=f"{wavelength_um * 1e3:.0f} nm",
        )
    if limit_um is not None:
        ax.set_xlim(-limit_um, limit_um)
        ax.set_ylim(-limit_um, limit_um)
    ax.set_aspect("equal")
    ax.set(xlabel="Δx (µm)", ylabel="Δy (µm)",
           title=title or f"Polychromatic spot — RMS {spots.polychromatic_rms_um:.2f} µm")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, markerscale=2, loc="upper right")
    fig.tight_layout()
    return fig


def zernike_figure(expansion: ZernikeExpansion) -> plt.Figure:
    """Coefficient bars plus nominal/refocused wavefront maps."""

    records = expansion.records
    coefficients = np.array([row["coefficient_nm_rms"] for row in records])
    labels = [
        f"Z{row['ansi_index']}\n({row['radial_order_n']},{row['azimuthal_frequency_m']})"
        for row in records
    ]
    grid = np.linspace(-1, 1, 301)
    uu, vv = np.meshgrid(grid, grid)
    disk = uu * uu + vv * vv <= 1
    nominal = expansion.wavefront(uu, vv, remove_tilt=True) * 1e6
    refocused = expansion.wavefront(uu, vv, remove_tilt=True, refocus=True) * 1e6
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
        (
            fig.add_subplot(grid_spec[1, 0]), nominal,
            f"Nominal plane: {expansion.rms_no_tilt_nm:.2f} nm RMS",
        ),
        (
            fig.add_subplot(grid_spec[1, 1]), refocused,
            f"After tilt/defocus removal: {expansion.rms_refocused_nm:.2f} nm RMS",
        ),
    ):
        limit = max(np.nanmax(np.abs(data)), 1e-6)
        shown = ax.imshow(
            data, extent=[-1, 1, -1, 1], origin="lower", cmap="RdBu_r",
            vmin=-limit, vmax=limit,
        )
        ax.set(xlabel="u", ylabel="v", title=title, aspect="equal")
        fig.colorbar(shown, ax=ax, label="Wavefront (nm)")
    fig.tight_layout()
    return fig


def wavefront_figure(samples: WavefrontSamples, *, title: str | None = None) -> plt.Figure:
    """Scattered exit-pupil OPD samples in milli-waves."""

    fig, ax = plt.subplots(figsize=(5, 4.4))
    shown = ax.scatter(samples.u, samples.v, c=samples.opd_waves * 1e3, cmap="RdBu_r", s=18)
    ax.set_aspect("equal")
    ax.set_xlabel("u (pupil)")
    ax.set_ylabel("v (pupil)")
    ax.set_title(title or "Exit-pupil wavefront W(u,v) [mλ]")
    fig.colorbar(shown, label="mλ")
    fig.tight_layout()
    return fig


def psf_figure(
    psf: np.ndarray, *, pixel_nm: float, crop: int = 55, title: str | None = None
) -> plt.Figure:
    """Central crop of the scalar PSF."""

    size = psf.shape[0]
    centre = size // 2
    coordinates_nm = (np.arange(size) - centre) * pixel_nm
    extent = [
        coordinates_nm[centre - crop], coordinates_nm[centre + crop - 1],
        coordinates_nm[centre - crop], coordinates_nm[centre + crop - 1],
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    shown = ax.imshow(
        psf[centre - crop : centre + crop, centre - crop : centre + crop],
        extent=extent, origin="lower", cmap="inferno",
    )
    ax.set(
        title=title or "Scalar PSF", xlabel="x at wafer (nm)", ylabel="y at wafer (nm)"
    )
    fig.colorbar(shown, ax=ax, label="Normalized intensity")
    fig.tight_layout()
    return fig


def aerial_figure(
    mask: np.ndarray,
    aerial: np.ndarray,
    *,
    pixel_nm: float,
    threshold: float = 0.30,
    title: str | None = None,
) -> plt.Figure:
    """Mask, Abbe aerial image, and thresholded resist panels."""

    size = mask.shape[0]
    x_nm = (np.arange(size) - size // 2) * pixel_nm
    extent = [x_nm[0], x_nm[-1], x_nm[0], x_nm[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(mask, extent=extent, origin="lower", cmap="gray_r")
    axes[0].set_title("Mask (wafer scale)")
    shown = axes[1].imshow(aerial, extent=extent, origin="lower", cmap="inferno")
    axes[1].set_title(title or "Abbe aerial image")
    fig.colorbar(shown, ax=axes[1], fraction=0.046)
    axes[2].imshow(aerial > threshold, extent=extent, origin="lower", cmap="gray_r")
    axes[2].set_title(f"Developed resist (threshold {threshold:.2f})")
    for ax in axes:
        ax.set_xlabel("x (nm)")
    axes[0].set_ylabel("y (nm)")
    fig.tight_layout()
    return fig


def contrast_figure(
    half_pitches_nm,
    contrasts,
    *,
    cutoff_nm: float | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Contrast vs half-pitch resolving-power curve."""

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(half_pitches_nm, contrasts, "o-", color="#1a6e3c", lw=1.8)
    if cutoff_nm is not None:
        ax.axvline(cutoff_nm, color="#b3421b", ls="--")
        ax.annotate(
            f"cutoff λ/[2·NA·(1+σ)] = {cutoff_nm:.1f} nm",
            (cutoff_nm, 0.66), color="#b3421b", ha="right", fontsize=9,
        )
    ax.axhline(0.30, color="#555", ls=":")
    hps = np.asarray(half_pitches_nm, dtype=float)
    ax.set_xlim(hps.max() * 1.05, hps.min() * 0.85)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.set_xlabel("half-pitch (nm)")
    ax.set_ylabel("image contrast")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


__all__ = [
    "layout_figure",
    "mirror_arcs_from_paths",
    "spots_figure",
    "fans_figure",
    "aberrations_figure",
    "distortion_grid_figure",
    "distortion_grids_figure",
    "dispersion_figure",
    "chromatic_figure",
    "chromatic_spots_figure",
    "zernike_figure",
    "wavefront_figure",
    "psf_figure",
    "aerial_figure",
    "contrast_figure",
]
