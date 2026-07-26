"""Pure numeric ray/shape intersection solvers.

Finding where a ray meets a shape is a mathematical root-finding problem,
nothing more: given a parametric line and an implicit or profile-defined
surface, solve for the parameter at which they coincide. What happens
afterward — deciding whether that hit is inside the clear aperture, what
new ray to spawn, whether to keep tracing — is propagation logic and lives
in :mod:`raytracer.propagation`, not here. Every function below takes raw
arrays/shape parameters and returns a distance (plus whatever the caller
needs to build a normal), never a ``Ray`` or ``Surface`` object.
"""

from __future__ import annotations

import numpy as np


def intersect_segment(
    origin: np.ndarray,
    direction: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    eps: float = 1e-14,
) -> tuple[np.ndarray, float, float] | None:
    """Nearest forward intersection of a 2-D ray with the segment ``p0 -> p1``.

    Returns ``(point, distance, s)`` where ``s`` in ``[0, 1]`` is the
    fraction along the segment, or ``None`` if the ray is parallel to the
    segment, misses it, or the intersection lies behind the ray origin.
    """

    d = direction
    e = p1 - p0
    denom = d[0] * (-e[1]) - d[1] * (-e[0])
    if abs(denom) < eps:
        return None
    rhs = p0 - origin
    t = (rhs[0] * (-e[1]) - rhs[1] * (-e[0])) / denom
    s = (d[0] * rhs[1] - d[1] * rhs[0]) / denom
    if t <= 1e-12 or s < 0.0 or s > 1.0:
        return None
    point = origin + t * d
    return point, float(t), float(s)


def intersect_quadratic(
    origin: np.ndarray,
    direction: np.ndarray,
    coeffs: tuple[float, float, float, float, float, float],
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, float] | None:
    """Nearest forward intersection of a 2-D ray with a quadratic curve.

    ``coeffs`` is ``(A, B, C, D, E, F)`` for ``A x^2 + B xy + C y^2 + D x +
    E y + F = 0`` (e.g. a conic section's quadratic form). Returns
    ``(point, distance)`` for the smallest non-negative root, or ``None``.
    """

    x0, y0 = origin
    dx, dy = direction
    A, B, C, D, E, F = coeffs

    a = A * dx * dx + B * dx * dy + C * dy * dy
    b = 2.0 * A * x0 * dx + B * (x0 * dy + y0 * dx) + 2.0 * C * y0 * dy + D * dx + E * dy
    c = A * x0 * x0 + B * x0 * y0 + C * y0 * y0 + D * x0 + E * y0 + F

    lam: float | None = None

    if abs(a) < eps:
        if abs(b) >= eps:
            candidate = -c / b
            if candidate >= 0.0:
                lam = candidate
    else:
        disc = b * b - 4.0 * a * c
        if disc >= -eps:
            disc = max(0.0, disc)
            sqrt_disc = np.sqrt(disc)
            lam1 = (-b - sqrt_disc) / (2.0 * a)
            lam2 = (-b + sqrt_disc) / (2.0 * a)
            candidates = [val for val in (lam1, lam2) if val >= 0.0]
            if candidates:
                lam = min(candidates)

    if lam is None:
        return None

    point = origin + lam * direction
    return point, float(lam)


def quadratic_normal(point: np.ndarray, coeffs: tuple[float, float, float, float, float, float]) -> np.ndarray:
    """Un-normalized gradient (surface normal direction) of the quadratic form at *point*."""

    x, y = point
    A, B, C, D, E, _ = coeffs
    nx = 2.0 * A * x + B * y + D
    ny = B * x + 2.0 * C * y + E
    return np.array([nx, ny], dtype=float)


def intersect_profile(
    origin: np.ndarray,
    direction: np.ndarray,
    profile,
    *,
    max_newton: int = 30,
    tol: float = 1e-12,
) -> tuple[float, float, float] | None:
    """Scalar Newton solve for a ray against a rotationally-symmetric profile.

    ``origin``/``direction`` are in the profile's own local frame (axis
    along +x, transverse coordinate the profile's ``h``). ``profile``
    exposes ``.sag_and_slope(h)`` (e.g. an ``AsphereProfile``). Returns
    ``(t, h, slope)`` at the first forward intersection, or ``None`` if the
    ray is parallel to a planar profile, the iteration fails to converge, or
    the intersection lies behind the ray origin.
    """

    o, d = origin, direction
    if abs(d[0]) < 1e-14 and profile.is_plane:
        return None

    t = (0.0 - o[0]) / d[0] if abs(d[0]) > 1e-14 else 0.0
    converged = False
    step = 0.0
    h = 0.0
    slope = 0.0
    for _ in range(max_newton):
        p = o + t * d
        h = abs(p[1])
        sag, slope = profile.sag_and_slope(h)
        f = p[0] - float(sag)
        signed_slope = float(slope) * np.sign(p[1])
        derivative = d[0] - signed_slope * d[1]
        if abs(derivative) < 1e-14:
            return None
        step = f / derivative
        t -= step
        if abs(step) < 1e-12:
            converged = True
            break
    if not converged and abs(step) > 1e-9:
        return None
    if t <= 1e-12:
        return None

    p = o + t * d
    h = abs(p[1])
    _, slope = profile.sag_and_slope(h)
    return float(t), float(h), float(slope)


def intersect_profile_batch(
    p: np.ndarray,
    d: np.ndarray,
    vertex_z: float,
    profile,
    alive: np.ndarray,
    *,
    max_newton: int = 15,
    newton_tol: float = 2e-11,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Newton solve for N rays against a rotationally-symmetric
    profile at axial position *vertex_z* (the exact 3-D sequential tracer's
    per-surface step).

    ``p``/``d`` are ``(N, 3)`` world-space ray origins/directions, ``alive``
    an ``(N,)`` mask of rays still worth solving for (already-failed rays
    are left alone). Returns ``(t, h, slope, parallel, diverged)``: ``t``,
    ``h`` (radial height at the hit) and ``slope`` are only meaningful where
    both ``parallel`` and ``diverged`` are ``False`` and the input ``alive``
    was ``True``. ``parallel`` marks rays running parallel to the surface's
    axis; ``diverged`` marks rays for which the Newton iteration could not
    proceed or failed to converge -- both are solve failures, not physics
    (vignetting/TIR are the caller's concern, decided from a *successful*
    ``t``/``h``).
    """

    n_rays = p.shape[0]
    dz = d[:, 2]
    parallel = alive & (np.abs(dz) < 1e-14)
    solving = alive & ~parallel
    diverged = np.zeros(n_rays, dtype=bool)

    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(solving, (vertex_z - p[:, 2]) / np.where(dz == 0.0, 1.0, dz), 0.0)
        step = np.zeros(n_rays)
        h = np.zeros(n_rays)
        slope = np.zeros(n_rays)
        for _ in range(max_newton):
            q = p + t[:, None] * d
            h = np.hypot(q[:, 0], q[:, 1])
            sag, slope = profile.sag_and_slope(h)
            residual = q[:, 2] - vertex_z - sag
            radial_dot = np.where(
                h > 0.0,
                (q[:, 0] * d[:, 0] + q[:, 1] * d[:, 1]) / np.where(h == 0.0, 1.0, h),
                0.0,
            )
            derivative = dz - slope * radial_dot
            stuck = solving & (np.abs(derivative) < 1e-14)
            if stuck.any():
                diverged |= stuck
                solving &= ~stuck
            step = np.where(solving, residual / np.where(derivative == 0.0, 1.0, derivative), 0.0)
            t = t - step
            if not np.any(np.abs(step[solving]) >= newton_tol):
                break

    diverged |= solving & (~np.isfinite(t) | (np.abs(step) > 1e-6))

    q = p + t[:, None] * d
    h = np.hypot(q[:, 0], q[:, 1])
    _, slope = profile.sag_and_slope(h)
    return t, h, slope, parallel, diverged


def intersect_fermat_oval(
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    z0: float,
    zi: float,
    n0: float,
    ni: float,
    implicit_form,
) -> tuple[np.ndarray, float] | None:
    """Bisection solve for a ray against a Cartesian oval's implicit form.

    ``implicit_form(z, r, z0, zi, n0, ni)`` is the surface's defining
    equation (see :func:`raytracer.shapes.fermat_oval.fermat_oval_F`);
    scans ``t`` in ``[0, 50]`` for a sign change, then bisects it down.
    Returns ``(point, distance)`` or ``None`` if no crossing is found.
    """

    def eval_at(t: float) -> float:
        pt = origin + t * direction
        return implicit_form(pt[0], abs(pt[1]), z0, zi, n0, ni)

    t_samples = np.linspace(0, 50, 500)
    F_samples = np.array([eval_at(t) for t in t_samples])

    sign_changes = np.where(np.diff(np.sign(F_samples)) != 0)[0]
    if len(sign_changes) == 0:
        return None

    idx = sign_changes[0]
    t_min = float(t_samples[idx])
    t_max = float(t_samples[idx + 1])

    t_mid = 0.5 * (t_min + t_max)
    for _ in range(100):
        t_mid = (t_min + t_max) / 2
        F_mid = eval_at(t_mid)

        if abs(F_mid) < 1e-10:
            return origin + t_mid * direction, float(t_mid)

        F_min = eval_at(t_min)

        if F_min * F_mid < 0:
            t_max = t_mid
        else:
            t_min = t_mid

        if abs(t_max - t_min) < 1e-12:
            break

    pt_final = origin + t_mid * direction
    F_final = eval_at(t_mid)
    if abs(F_final) < 1e-6:
        return pt_final, float(t_mid)
    return None


__all__ = [
    "intersect_segment",
    "intersect_quadratic",
    "quadratic_normal",
    "intersect_profile",
    "intersect_profile_batch",
    "intersect_fermat_oval",
]
