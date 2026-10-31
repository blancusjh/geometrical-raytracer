"""Matplotlib analysis figures of sequential systems; each returns its :class:`~matplotlib.figure.Figure`."""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..analysis.aberrations.chromatic import AxialColor, ChromaticSpots, LateralColor
from ..analysis.aberrations.distortion import DistortionGrid
from ..analysis.aberrations.fans import FanData
from ..analysis.aberrations.wavefront import WavefrontSamples
from ..analysis.aberrations.zernike import ZernikeExpansion
from ..analysis.imaging.spots import SpotData
from ..rays import Rays
from ..tracing import FieldPoint, SequentialTracer, chief_ray_slopes, trace_from_object
from .bodies import DEFAULT_MATERIAL_COLORS, draw_system, draw_traced

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
    """Meridional layout of a sequential trace: bodies, mirrors, the object, image and stop planes, ray fans.

    Each field is a fan of slopes ``±fan_half_slope`` about its chief ray. Bodies are cut apart where they
    interpenetrate and coloured by index (:func:`~raytracer.viz.bodies.body_outlines`), or by medium name
    through ``material_colors`` over ``DEFAULT_MATERIAL_COLORS``; mirrors span the sub-aperture the fans use,
    labelled by name; each field's foci are marked in its colour.
    """

    system = tracer.system
    fig, ax = plt.subplots(figsize=figsize)
    beams = []
    for field_y, color in zip(fields, FIELD_COLORS):
        _, chief = chief_ray_slopes(tracer, FieldPoint(y=field_y))
        slopes = chief + np.linspace(-fan_half_slope, fan_half_slope, fan_count)
        beams.append((color, Rays.concatenate([trace_from_object(tracer, (0.0, field_y), (0.0, s)) for s in slopes])))
    draw_traced(ax, tracer, beams, draw_failed=True, ray_style=dict(lw=0.65, alpha=0.72, zorder=3),
                material_colors={**DEFAULT_MATERIAL_COLORS, **(material_colors or {})}, alpha=0.66,
                mirror_labels=mirror_labels, sensors=False)

    if tracer.object_z is not None:
        ax.axvline(tracer.object_z, color="black", ls="--", lw=1, label="object plane")
    ax.axvline(tracer.image_z, color="black", ls="--", lw=1, label="image plane")
    if system.stop_index is not None:
        ax.axvline(tracer.step_z(system.stop_index), color="#555555", ls=":", lw=1, label="aperture stop")
    ax.axhline(0, color="#aaaaaa", lw=0.6)
    ax.set(xlabel="Global z (mm)", ylabel="Meridional y (mm)", title=title or (system.name or "Sequential layout"))
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


def spots_figure(spots: Iterable[SpotData], *, airy_radius_um=None, figsize_per_panel=4.0, colors=None, titles=None) -> plt.Figure:
    """Spot diagrams about the weighted centroid, one panel per field.

    ``colors=FIELD_COLORS`` keeps each field's colour from :func:`layout_figure`; ``titles`` head the panels
    (default: the object height), above the RMS line.
    """

    spots = list(spots)
    fig, axes = plt.subplots(1, len(spots), figsize=(figsize_per_panel * len(spots), figsize_per_panel), squeeze=False)
    for k, (spot, ax, color) in enumerate(zip(spots, axes[0], colors if colors is not None else [None] * len(spots))):
        ax.scatter(spot.relative_um[:, 0], spot.relative_um[:, 1], s=4, alpha=0.55, color=color)
        if airy_radius_um is not None:
            ax.add_patch(plt.Circle((0, 0), airy_radius_um, fill=False, color="#d62728", ls="--", lw=1.0))
        head = titles[k] if titles is not None else f"Object y={spot.field_y:.0f} mm"
        ax.set_title(f"{head}\narea-weighted RMS={spot.rms_radius_um:.4g} µm")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_xlabel("Δx (µm)")
    axes[0, 0].set_ylabel("Δy (µm)")
    fig.suptitle("Geometric spot diagrams — unvignetted rays")
    fig.tight_layout()
    return fig


def fans_figure(fans: Iterable[FanData]) -> plt.Figure:
    """Tangential and sagittal transverse-aberration fans."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for fan, color in zip(fans, FIELD_COLORS):
        axes[0].plot(fan.tangential[:, 0], fan.tangential[:, 1], color=color, label=f"y={fan.field_y:.0f} mm")
        axes[1].plot(fan.sagittal[:, 0], fan.sagittal[:, 1], color=color, label=f"y={fan.field_y:.0f} mm")
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
    """2×2 field summary: RMS spot, distortion, field curvature, astigmatism."""

    h = np.array([row["centroid_image_height_mm"] for row in metrics])

    def column(key):
        return [row[key] for row in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    (spot, distortion), (curvature, astigmatism) = axes
    spot.plot(h, column("rms_spot_radius_um"), "o-", label="nominal plane")
    spot.plot(h, column("best_focus_rms_spot_um"), "s-", label="best focus")
    spot.set(ylabel="RMS radius (µm)", title="Transverse aberration")
    distortion.plot(h, column("chief_ray_distortion_um"), "o-")
    distortion.set(ylabel="Chief-ray distortion (µm)", title="Distortion")
    curvature.plot(h, column("sagittal_focus_shift_mm"), "o-", label="sagittal")
    curvature.plot(h, column("tangential_focus_shift_mm"), "s-", label="tangential")
    curvature.set(ylabel="Best-focus shift (mm)", title="Field curvature")
    astigmatism.plot(h, column("astigmatic_separation_mm"), "o-")
    astigmatism.set(ylabel="T−S focus (mm)", title="Astigmatism")
    for ax in (distortion, curvature, astigmatism):
        ax.axhline(0, color="black", lw=0.7)
    for ax in (spot, curvature):
        ax.legend()
    for ax in axes.flat:
        ax.set_xlabel("Image height (mm)")
        ax.grid(alpha=0.2)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def distortion_grid_figure(grid: DistortionGrid, *, exaggeration=1.0, title=None, figsize=(6.5, 6.5)) -> plt.Figure:
    """A traced square grid over the ideal one, its departure from ideal amplified by ``exaggeration``.

    Real distortion is usually too small to see at 1:1, so the dial is adjustable, as optical design software
    offers it. Vignetted points break their lines instead of being interpolated over, and are marked ``x``.
    """

    fig, ax = plt.subplots(figsize=figsize)
    _draw_distortion_grid(ax, grid, exaggeration=exaggeration)
    ax.set_ylabel("image y (mm)")
    ax.set_title((title or "Distortion grid") + (f" (distortion ×{exaggeration:g})" if exaggeration != 1.0 else ""))
    fig.tight_layout()
    return fig


def distortion_grids_figure(grids: Iterable[DistortionGrid], *, titles=None, exaggerations=1.0, figsize_per_panel=4.4,
                            suptitle=None) -> plt.Figure:
    """Several distortion grids side by side, each with its own exaggeration (one value, or one per grid).

    Systems worth comparing span orders of magnitude in distortion, so each panel title carries the true
    magnitude — the largest departure, in µm and as a fraction of image height — next to its amplification.
    """

    grids = list(grids)
    titles = titles if titles is not None else [""] * len(grids)
    if isinstance(exaggerations, (int, float)):
        exaggerations = [float(exaggerations)] * len(grids)
    fig, axes = plt.subplots(1, len(grids), figsize=(figsize_per_panel * len(grids), figsize_per_panel + 0.6), squeeze=False)
    for grid, ax, title, exaggeration in zip(grids, axes[0], titles, exaggerations):
        _draw_distortion_grid(ax, grid, exaggeration=exaggeration)
        ax.set_title(f"{title}\nmax {grid.max_distortion_um:.3g} µm = {grid.max_relative_distortion_percent:.3g} % of height"
                     f"   —   drawn ×{exaggeration:g}", fontsize=10)
    axes[0, 0].set_ylabel("image y (mm)")
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def _draw_distortion_grid(ax, grid: DistortionGrid, *, exaggeration: float, color: str = "#1f77b4") -> None:
    """The ideal grid dashed, the traced one solid over it (vignetted points are NaN, so their lines break)."""

    ideal = grid.ideal_points
    displayed = ideal + exaggeration * (grid.actual_points - ideal)
    for i in range(ideal.shape[0]):
        for line in (np.s_[i, :], np.s_[:, i]):
            ax.plot(ideal[line][:, 0], ideal[line][:, 1], color="#999999", ls="--", lw=0.7, zorder=1)
            ax.plot(displayed[line][:, 0], displayed[line][:, 1], color=color, lw=1.4, zorder=2)
    if not grid.valid.all():
        dropped = ideal[~grid.valid]
        ax.scatter(dropped[:, 0], dropped[:, 1], marker="x", color="#d62728", s=24, zorder=3,
                   label=f"vignetted ({int((~grid.valid).sum())}/{grid.valid.size})")
        ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel("image x (mm)")
    ax.grid(alpha=0.15)


def dispersion_figure(materials: Sequence, wavelengths_um: np.ndarray, *, reference_lines_um=(), title=None) -> plt.Figure:
    """Refractive index against wavelength; the legend names each material's model (Sellmeier, Abbe, …)."""

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for material in materials:
        ax.plot(wavelengths_um, [material.index(float(w)) for w in wavelengths_um], "-", lw=1.6,
                label=f"{material.name} ({type(material).__name__})")
    for wavelength_um in reference_lines_um:
        ax.axvline(wavelength_um, color="#999999", ls=":", lw=0.8)
    ax.set(xlabel="wavelength (µm)", ylabel="refractive index n", title=title or "Glass dispersion")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def chromatic_figure(axial: AxialColor, lateral: LateralColor | None = None, *, reference_lines_um=(), suptitle=None) -> plt.Figure:
    """Axial colour, and optionally lateral colour, against wavelength."""

    panels = 1 if lateral is None else 2
    fig, axes = plt.subplots(1, panels, figsize=(5.5 * panels, 4.2), squeeze=False)
    axes = axes[0]
    axes[0].plot(axial.wavelengths_um, axial.focus_shift_mm * 1e3, "-", color="#b3421b", lw=1.6)
    axes[0].set(xlabel="wavelength (µm)", ylabel="focus shift (µm)", title="Axial color")
    if lateral is not None:
        axes[1].plot(lateral.wavelengths_um, lateral.lateral_color_um, "-", color="#2878b5", lw=1.6)
        axes[1].set(xlabel="wavelength (µm)", ylabel="image height shift (µm)",
                    title=f"Lateral color at y={lateral.field_height_mm:.4g} mm")
    for ax in axes:
        ax.axhline(0, color="black", lw=0.7)
        for wavelength_um in reference_lines_um:
            ax.axvline(wavelength_um, color="#999999", ls=":", lw=0.8)
        ax.grid(alpha=0.2)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def chromatic_spots_figure(spots: ChromaticSpots, *, limit_um=None, title=None, figsize=(6.0, 6.0)) -> plt.Figure:
    """Every wavelength's spot in one reference frame, coloured blue to red across the spectrum.

    ``limit_um`` holds the axes fixed across figures, or a tight spot and a blurred one look alike.
    """

    wavelengths = spots.wavelengths_um
    lo, hi = float(wavelengths.min()), float(wavelengths.max())
    fig, ax = plt.subplots(figsize=figsize)
    for wavelength_um, offsets in zip(wavelengths, spots.offsets_um):
        fraction = 0.5 if hi == lo else (wavelength_um - lo) / (hi - lo)
        # turbo runs blue to red like the spectrum; its near-black ends are left out
        ax.scatter(offsets[:, 0], offsets[:, 1], s=5, alpha=0.6, color=plt.get_cmap("turbo")(0.06 + 0.88 * fraction),
                   label=f"{wavelength_um * 1e3:.0f} nm")
    if limit_um is not None:
        ax.set_xlim(-limit_um, limit_um)
        ax.set_ylim(-limit_um, limit_um)
    ax.set_aspect("equal")
    ax.set(xlabel="Δx (µm)", ylabel="Δy (µm)", title=title or f"Polychromatic spot — RMS {spots.polychromatic_rms_um:.2f} µm")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, markerscale=2, loc="upper right")
    fig.tight_layout()
    return fig


def zernike_figure(expansion: ZernikeExpansion) -> plt.Figure:
    """Coefficient bars, and the wavefront at the nominal plane and refocused."""

    records = expansion.records
    labels = [f"Z{r['ansi_index']}\n({r['radial_order_n']},{r['azimuthal_frequency_m']})" for r in records]
    uu, vv = np.meshgrid(np.linspace(-1, 1, 301), np.linspace(-1, 1, 301))
    outside = uu * uu + vv * vv > 1

    fig = plt.figure(figsize=(14, 8))
    grid_spec = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    bars = fig.add_subplot(grid_spec[0, :])
    bars.bar(np.arange(len(records)), [r["coefficient_nm_rms"] for r in records], color="#2878b5")
    bars.set_xticks(np.arange(len(records)), labels, rotation=70, ha="right")
    bars.set(ylabel="RMS coefficient (nm)", title="ANSI/OSA Zernike coefficients")
    bars.grid(axis="y", alpha=0.2)
    for column, refocus, title in (
        (0, False, f"Nominal plane: {expansion.rms_no_tilt_nm:.2f} nm RMS"),
        (1, True, f"After tilt/defocus removal: {expansion.rms_refocused_nm:.2f} nm RMS"),
    ):
        ax = fig.add_subplot(grid_spec[1, column])
        data = expansion.wavefront(uu, vv, remove_tilt=True, refocus=refocus) * 1e6
        data[outside] = np.nan
        limit = max(np.nanmax(np.abs(data)), 1e-6)
        shown = ax.imshow(data, extent=[-1, 1, -1, 1], origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set(xlabel="u", ylabel="v", title=title, aspect="equal")
        fig.colorbar(shown, ax=ax, label="Wavefront (nm)")
    fig.tight_layout()
    return fig


def wavefront_figure(samples: WavefrontSamples, *, title: str | None = None) -> plt.Figure:
    """Scattered exit-pupil OPD samples in milli-waves."""

    fig, ax = plt.subplots(figsize=(5, 4.4))
    shown = ax.scatter(samples.u, samples.v, c=samples.opd_waves * 1e3, cmap="RdBu_r", s=18)
    ax.set(aspect="equal", xlabel="u (pupil)", ylabel="v (pupil)", title=title or "Exit-pupil wavefront W(u,v) [mλ]")
    fig.colorbar(shown, label="mλ")
    fig.tight_layout()
    return fig


def psf_figure(psf: np.ndarray, *, pixel_nm: float, crop: int = 55, title: str | None = None) -> plt.Figure:
    """Central crop of the scalar PSF."""

    centre = psf.shape[0] // 2
    nm = (np.arange(psf.shape[0]) - centre) * pixel_nm
    lo, hi = nm[centre - crop], nm[centre + crop - 1]
    fig, ax = plt.subplots(figsize=(6, 5))
    shown = ax.imshow(psf[centre - crop : centre + crop, centre - crop : centre + crop], extent=[lo, hi, lo, hi],
                      origin="lower", cmap="inferno")
    ax.set(title=title or "Scalar PSF", xlabel="x at wafer (nm)", ylabel="y at wafer (nm)")
    fig.colorbar(shown, ax=ax, label="Normalized intensity")
    fig.tight_layout()
    return fig


def aerial_figure(mask: np.ndarray, aerial: np.ndarray, *, pixel_nm: float, threshold=0.30, title=None) -> plt.Figure:
    """Mask, Abbe aerial image, and thresholded resist panels."""

    x_nm = (np.arange(mask.shape[0]) - mask.shape[0] // 2) * pixel_nm
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


def contrast_figure(half_pitches_nm, contrasts, *, cutoff_nm: float | None = None, title: str | None = None) -> plt.Figure:
    """Image contrast against half-pitch: the resolving-power curve."""

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(half_pitches_nm, contrasts, "o-", color="#1a6e3c", lw=1.8)
    if cutoff_nm is not None:
        ax.axvline(cutoff_nm, color="#b3421b", ls="--")
        ax.annotate(f"cutoff λ/[2·NA·(1+σ)] = {cutoff_nm:.1f} nm", (cutoff_nm, 0.66), color="#b3421b", ha="right", fontsize=9)
    ax.axhline(0.30, color="#555", ls=":")
    hps = np.asarray(half_pitches_nm, dtype=float)
    ax.set_xlim(hps.max() * 1.05, hps.min() * 0.85)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.set(xlabel="half-pitch (nm)", ylabel="image contrast")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


__all__ = [
    "draw_system",
    "layout_figure",
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
