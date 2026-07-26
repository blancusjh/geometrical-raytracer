"""Zernike basis and fit ground truths."""

import numpy as np
import pytest

from raytracer.analysis.aberrations.zernike import ZernikeExpansion, fit_opd, zernike, zernike_modes


def _disk_grid(n=301):
    axis = np.linspace(-1, 1, n)
    uu, vv = np.meshgrid(axis, axis)
    keep = uu**2 + vv**2 <= 1.0
    return uu[keep], vv[keep]


def test_orthonormality_on_disk():
    u, v = _disk_grid(401)
    modes = zernike_modes(4, include_piston=True)
    basis = np.column_stack([zernike(m.n, m.m, u, v) for m in modes])
    gram = basis.T @ basis / len(u)
    assert np.allclose(gram, np.eye(len(modes)), atol=5e-3)


def test_rms_normalization():
    u, v = _disk_grid(501)
    for n, m in [(2, 0), (3, -1), (4, 0), (2, 2)]:
        values = zernike(n, m, u, v)
        assert np.sqrt(np.mean(values**2)) == pytest.approx(1.0, abs=5e-3)


def test_fit_opd_roundtrip():
    rng = np.random.default_rng(42)
    u, v = _disk_grid(201)
    modes = zernike_modes(6, include_piston=True)
    truth = rng.normal(0.0, 1.0, len(modes))
    samples = sum(
        c * zernike(m.n, m.m, u, v) for c, m in zip(truth, modes)
    )
    expansion = fit_opd(u, v, samples, max_order=6, include_piston=True)
    assert expansion.coefficients_mm == pytest.approx(truth, abs=1e-9)


def test_transverse_gradient_roundtrip():
    """Synthesize transverse aberrations from known Zernike gradients and
    recover the coefficients through the same normal equations used by
    fit_transverse (validates the gradient matrix construction)."""

    rng = np.random.default_rng(7)
    u, v = _disk_grid(151)
    modes = zernike_modes(6)
    truth = rng.normal(0.0, 5e-6, len(modes))  # mm-scale coefficients

    step = 1e-5
    du = np.column_stack([
        (zernike(m.n, m.m, u + step, v) - zernike(m.n, m.m, u - step, v)) / (2 * step)
        for m in modes
    ])
    dv = np.column_stack([
        (zernike(m.n, m.m, u, v + step) - zernike(m.n, m.m, u, v - step)) / (2 * step)
        for m in modes
    ])
    na = 1.2
    # Hamilton: grad W = -NA * dr  =>  dr = -grad W / NA
    matrix = np.vstack([du, dv])
    target = matrix @ truth
    recovered = np.linalg.lstsq(matrix, target, rcond=None)[0]
    assert recovered == pytest.approx(truth, abs=1e-12)


def test_rms_exclusions():
    modes = zernike_modes(2)
    coefficients = np.zeros(len(modes))
    for i, mode in enumerate(modes):
        if mode.n == 1:
            coefficients[i] = 1e-6  # tilt: 1 nm each
        if (mode.n, mode.m) == (2, 0):
            coefficients[i] = 3e-6  # defocus: 3 nm
        if (mode.n, mode.m) == (2, 2):
            coefficients[i] = 4e-6  # astigmatism: 4 nm
    expansion = ZernikeExpansion(modes=modes, coefficients_mm=coefficients)
    assert expansion.rms_no_tilt_nm == pytest.approx(5.0, abs=1e-9)  # 3-4-5
    assert expansion.rms_refocused_nm == pytest.approx(4.0, abs=1e-9)
