"""Eikonal wavefront and scalar-imaging ground truths."""

import numpy as np
import pytest

from raytracer.analysis.imaging import (
    airy_radius_mm,
    coherent_cutoff_half_pitch_nm,
    contrast_curve,
    pupil_function,
    scalar_psf,
)
from raytracer.analysis.aberrations.wavefront import exit_pupil_wavefront
from raytracer.optics.materials import AIR, ConstantIndex
from raytracer.design import OpticalSystem, SurfaceRow
from raytracer.propagation import FieldPoint, PupilSampling, SequentialTracer, trace_pupil


def test_eikonal_opd_is_zero_for_stigmatic_conjugates():
    a, b = 1000.0, 600.0
    c = np.sqrt(a * a - b * b)
    near, far = -(a - c), -(a + c)
    # Object at the near focus; single mirror at z=0; image plane at far focus.
    rows = [SurfaceRow.mirror(radius=-(b * b) / a, thickness=far, conic=-((c / a) ** 2))]
    system = OpticalSystem(rows)
    system.object_z = near
    tracer = SequentialTracer(system, restart_offset=0.0)

    pupil = trace_pupil(
        tracer,
        FieldPoint(y=0.0),
        na_object_sine=0.2,
        sampling=PupilSampling(kind="rings", radial=8, azimuth=24),
        slope_model="sine",
        chief_slope=0.0,  # on-axis stigmatic point: chief is the axis itself
    )
    wavelength_mm = 13.4e-6
    samples = exit_pupil_wavefront(
        pupil, na_image=0.5, wavelength_mm=wavelength_mm, reference_radius=300.0,
        remove_mean=True,
    )
    # Perfect conjugates: OPD constant, i.e. ~0 after mean removal.
    assert np.max(np.abs(samples.opd_waves)) < 1e-6


def test_cartesian_oval_3d_is_stigmatic():
    """The 3-D GOTS Cartesian oval (SurfaceRow.cartesian_oval) must give OPD ~ 0,
    same as the 2-D implicit ovoid it shares its math with."""

    n0, z0, ni, zi = 1.0, -30.0, 1.7, 10.0
    glass = ConstantIndex("TEST_GLASS", ni)
    row = SurfaceRow.cartesian_oval(
        n0=n0, z0=z0, ni=ni, zi=zi, thickness=zi, material=glass, semidiameter=3.0
    )
    system = OpticalSystem([row], object_space=AIR, object_z=z0)
    tracer = SequentialTracer(system, restart_offset=0.0)

    pupil = trace_pupil(
        tracer, FieldPoint(y=0.0), na_object_sine=0.1,
        sampling=PupilSampling(kind="rings", radial=8, azimuth=24),
        slope_model="sine", chief_slope=0.0,
    )
    samples = exit_pupil_wavefront(
        pupil, na_image=0.3, wavelength_mm=0.5e-3, reference_radius=5.0, remove_mean=True,
    )
    assert np.max(np.abs(samples.opd_waves)) < 1e-6


def test_flat_pupil_airy_first_zero():
    na = 0.22
    wavelength_mm = 13.4e-6
    pixel_mm = 3.0e-6
    grid = pupil_function(None, na=na, wavelength_mm=wavelength_mm, size=512, pixel_mm=pixel_mm)
    psf = scalar_psf(grid)
    centre = 256
    profile = psf[centre, centre:]
    # First minimum of the sampled Airy pattern vs 0.61 lambda/NA.
    minima = np.where((profile[1:-1] < profile[:-2]) & (profile[1:-1] < profile[2:]))[0]
    first_zero_mm = (minima[0] + 1) * pixel_mm
    expected = airy_radius_mm(na, wavelength_mm)
    assert first_zero_mm == pytest.approx(expected, abs=pixel_mm)


def test_contrast_collapses_at_partial_coherence_cutoff():
    na = 0.22
    wavelength_mm = 13.4e-6
    sigma = 0.7
    cutoff = coherent_cutoff_half_pitch_nm(na, wavelength_mm, sigma)  # ~17.9 nm
    hps = np.array([2.0 * cutoff, 1.2 * cutoff, 0.85 * cutoff])
    contrasts = contrast_curve(
        None, hps, na=na, wavelength_mm=wavelength_mm, sigma=sigma,
        size=2048, pixel_nm=1.0, source_points=13,
    )
    assert contrasts[0] > 0.6  # comfortably resolved
    assert contrasts[1] > contrasts[2]
    # Beyond the cutoff the modulation collapses (small residual comes from
    # the discretized source integration).
    assert contrasts[2] < 0.15 * contrasts[0]


def test_defocused_reference_sphere_gives_defocus():
    """Displacing the image plane of a perfect system produces a wavefront
    dominated by Z(2,0) defocus."""

    from raytracer.analysis.aberrations.zernike import fit_opd

    a, b = 1000.0, 600.0
    c = np.sqrt(a * a - b * b)
    near, far = -(a - c), -(a + c)
    defocus_mm = 0.5
    rows = [SurfaceRow.mirror(radius=-(b * b) / a, thickness=far + defocus_mm, conic=-((c / a) ** 2))]
    system = OpticalSystem(rows)
    system.object_z = near
    tracer = SequentialTracer(system, restart_offset=0.0)

    pupil = trace_pupil(
        tracer, FieldPoint(y=0.0), na_object_sine=0.15,
        sampling=PupilSampling(kind="rings", radial=10, azimuth=32),
        slope_model="sine", chief_slope=0.0,
    )
    wavelength_mm = 13.4e-6
    samples = exit_pupil_wavefront(
        pupil, na_image=0.4, wavelength_mm=wavelength_mm, reference_radius=300.0
    )
    expansion = fit_opd(
        samples.u, samples.v, samples.opd_waves, max_order=4, include_piston=True,
        wavelength_mm=wavelength_mm,
    )
    coefficients = np.abs(expansion.coefficients_mm)
    defocus_idx = next(
        i for i, m in enumerate(expansion.modes) if (m.n, m.m) == (2, 0)
    )
    # Defocus dominates every other non-piston mode by an order of magnitude.
    others = [
        c for i, (c, m) in enumerate(zip(coefficients, expansion.modes))
        if i != defocus_idx and m.n > 0
    ]
    assert coefficients[defocus_idx] > 10 * max(others)
