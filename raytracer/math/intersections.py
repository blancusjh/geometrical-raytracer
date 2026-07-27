"""Where a ray meets a surface: one problem, two methods.

A ray is the set of points ``A + lambda u``. A surface describes itself in
one of two ways, and each gives a way to intersect it::

    implicit    f_Sigma(A + lambda u) = 0   ->  smallest lambda > 0
    parametric  A + lambda u = P(t)         ->  smallest lambda > 0

:func:`intersect_ray_with_surface` is the single entry point: it asks the
surface which description it offers and applies the matching method.

Within the implicit method the solver takes the cheapest exact route the
surface allows — closed form when ``f_Sigma`` is quadratic, Newton from a
surface-supplied seed, bracket-and-bisect otherwise. Those are three ways
of solving *the same equation*, not three methods.
"""

from __future__ import annotations

import numpy as np

MIN_LAMBDA = 1e-12


def intersect_ray_with_surface(origin, direction, surface):
    """Smallest ``lambda > 0`` at which ``A + lambda u`` meets *surface*.

    ``origin``/``direction`` are in the surface's own local frame (the
    surface converts, since the frame is part of how it is placed, not part
    of the math). Returns ``(point, lambda)`` or ``None`` if the ray misses.

    The method used follows ``surface.intersection_method``:
    ``"implicit"`` solves ``f_Sigma(A + lambda u) = 0``; ``"parametric"``
    solves ``A + lambda u = P(t)``.
    """

    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    method = getattr(surface, "intersection_method", "implicit")

    if method == "implicit":
        return _solve_implicit(origin, direction, surface)
    if method == "parametric":
        return _solve_parametric(origin, direction, surface)
    raise ValueError(
        f"{type(surface).__name__}.intersection_method must be 'implicit' or "
        f"'parametric', got {method!r}"
    )


# ---------------------------------------------------------------------------
# Method 1: implicit --  f_Sigma(A + lambda u) = 0
# ---------------------------------------------------------------------------


def _solve_implicit(origin, direction, surface):
    """Dispatch to the cheapest exact solve of ``f_Sigma(A + lambda u) = 0``."""

    quadratic = getattr(surface, "quadratic_form", None)
    if quadratic is not None:
        return solve_implicit_quadratic(origin, direction, quadratic)

    seed = surface.newton_seed(origin, direction)
    if seed is not None:
        return solve_implicit_newton(origin, direction, surface, seed)

    return solve_implicit_bracket(origin, direction, surface, *surface.lambda_search)


def solve_implicit_quadratic(origin, direction, coeffs):
    """Closed-form roots when ``f_Sigma`` is quadratic (2-D).

    ``coeffs`` is ``(A, B, C, D, E, F)`` for
    ``A x^2 + B xy + C y^2 + D x + E y + F = 0``. Substituting the ray makes
    this ``a lambda^2 + b lambda + c = 0``; a degenerate ``a == 0`` (a line,
    or a ray tangent to the axis of a parabola) falls back to the linear
    root. Returns the smallest non-negative root.
    """

    x0, y0 = origin
    dx, dy = direction
    A, B, C, D, E, F = coeffs
    eps = 1e-12

    a = A * dx * dx + B * dx * dy + C * dy * dy
    b = 2.0 * A * x0 * dx + B * (x0 * dy + y0 * dx) + 2.0 * C * y0 * dy + D * dx + E * dy
    c = A * x0 * x0 + B * x0 * y0 + C * y0 * y0 + D * x0 + E * y0 + F

    lam = None
    if abs(a) < eps:
        if abs(b) >= eps:
            candidate = -c / b
            if candidate >= 0.0:
                lam = candidate
    else:
        disc = b * b - 4.0 * a * c
        if disc >= -eps:
            sqrt_disc = np.sqrt(max(0.0, disc))
            roots = [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]
            forward = [r for r in roots if r >= 0.0]
            if forward:
                lam = min(forward)

    if lam is None:
        return None
    return origin + lam * direction, float(lam)


def solve_implicit_newton(origin, direction, surface, seed, *, max_iter=30, tol=1e-12):
    """Newton on ``g(lambda) = f_Sigma(A + lambda u)``, from *seed*.

    ``g'(lambda) = grad f_Sigma(A + lambda u) . u``, so the surface's own
    implicit gradient — the same one that gives the surface normal — drives
    the iteration. Returns ``None`` when the derivative vanishes (the ray
    grazes the surface) or the iteration does not converge.
    """

    lam = float(seed)
    step = 0.0
    converged = False
    for _ in range(max_iter):
        point = origin + lam * direction
        derivative = float(np.dot(surface.implicit_gradient(point), direction))
        if abs(derivative) < 1e-14:
            return None
        step = float(surface.implicit(point)) / derivative
        lam -= step
        if abs(step) < tol:
            converged = True
            break
    if not converged and abs(step) > 1e-9:
        return None
    if lam <= MIN_LAMBDA:
        return None
    return origin + lam * direction, float(lam)


def solve_implicit_bracket(origin, direction, surface, lam_max=50.0, samples=500,
                           *, max_iter=100, tol=1e-10):
    """Scan ``[0, lam_max]`` for the first sign change of ``f_Sigma``, then bisect.

    The robust fallback for an implicit form with no good Newton seed: it
    cannot be fooled by a bad initial guess, and by taking the *first* sign
    change it returns the smallest root by construction.
    """

    def g(lam):
        return float(surface.implicit(origin + lam * direction))

    lambdas = np.linspace(0.0, lam_max, samples)
    values = np.array([g(lam) for lam in lambdas])
    crossings = np.where(np.diff(np.sign(values)) != 0)[0]
    if crossings.size == 0:
        return None

    lo = float(lambdas[crossings[0]])
    hi = float(lambdas[crossings[0] + 1])
    mid = 0.5 * (lo + hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        g_mid = g(mid)
        if abs(g_mid) < tol:
            return origin + mid * direction, float(mid)
        if g(lo) * g_mid < 0.0:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) < 1e-12:
            break

    if abs(g(mid)) < 1e-6:
        return origin + mid * direction, float(mid)
    return None


# ---------------------------------------------------------------------------
# Method 2: parametric --  A + lambda u = P(t)
# ---------------------------------------------------------------------------


def solve_parametric(origin, direction, surface, seed, *, max_iter=60, tol=1e-13):
    """Newton on the 2-D system ``R(lambda, t) = A + lambda u - P(t) = 0``.

    The Jacobian is ``[u, -P'(t)]``; ``P'`` comes from
    ``surface.parametric_derivative`` when the surface provides it, and from
    a central difference otherwise. *seed* is the ``(lambda, t)`` starting
    guess the surface supplies.
    """

    state = np.array(seed, dtype=float)
    for _ in range(max_iter):
        lam, t = float(state[0]), float(state[1])
        residual = origin + lam * direction - surface.parametric(t)
        if np.max(np.abs(residual)) < tol:
            break
        jacobian = np.column_stack([direction, -_parametric_derivative(surface, t)])
        try:
            delta = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            return None
        state = state + delta

    lam, t = float(state[0]), float(state[1])
    point = surface.parametric(t)
    if not np.all(np.isfinite(point)):
        return None
    if np.max(np.abs(origin + lam * direction - point)) > 1e-7:
        return None
    if lam <= MIN_LAMBDA:
        return None
    return point, lam


def _parametric_derivative(surface, t, *, eps=1e-7):
    explicit = getattr(surface, "parametric_derivative", None)
    if explicit is not None:
        return np.asarray(explicit(t), dtype=float)
    step = eps * max(abs(t), 1.0)
    return (surface.parametric(t + step) - surface.parametric(t - step)) / (2.0 * step)


def _solve_parametric(origin, direction, surface):
    return solve_parametric(origin, direction, surface, surface.parametric_seed(origin, direction))


# ---------------------------------------------------------------------------
# Batch: the implicit method, vectorized over N rays against one surface
# ---------------------------------------------------------------------------


def intersect_rays_with_profile_surface(
    p: np.ndarray,
    d: np.ndarray,
    vertex_z: float,
    profile,
    alive: np.ndarray,
    *,
    max_newton: int = 15,
    newton_tol: float = 2e-11,
):
    """The implicit method, vectorized: N rays against one rotationally-symmetric
    profile surface at axial position *vertex_z*.

    Same equation and same Newton iteration as
    :func:`solve_implicit_newton`, written out over ``(N, 3)`` arrays so the
    sequential propagation can advance a whole bundle per surface. The
    implicit form of a profile surface of revolution about z is
    ``f_Sigma(x, y, z) = z - vertex_z - sag(hypot(x, y))``, whose gradient is
    ``(-slope x/h, -slope y/h, 1)``; the Newton derivative below is that
    gradient dotted with the ray direction.

    ``alive`` masks rays still worth solving for. Returns
    ``(lambda, h, slope, parallel, diverged)``; ``lambda``/``h``/``slope`` are
    meaningful only where neither failure mask is set. ``parallel`` marks
    rays running perpendicular to the axis, ``diverged`` marks a failed
    solve — both are *solve* failures. Vignetting and TIR are physics and
    stay with the caller.
    """

    n_rays = p.shape[0]
    dz = d[:, 2]
    parallel = alive & (np.abs(dz) < 1e-14)
    solving = alive & ~parallel
    diverged = np.zeros(n_rays, dtype=bool)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Seed: where the ray crosses the surface's vertex plane.
        t = np.where(solving, (vertex_z - p[:, 2]) / np.where(dz == 0.0, 1.0, dz), 0.0)
        step = np.zeros(n_rays)
        for _ in range(max_newton):
            q = p + t[:, None] * d
            h = np.hypot(q[:, 0], q[:, 1])
            sag, slope = profile.sag_and_slope(h)
            residual = q[:, 2] - vertex_z - sag  # f_Sigma(A + lambda u)
            radial_dot = np.where(
                h > 0.0,
                (q[:, 0] * d[:, 0] + q[:, 1] * d[:, 1]) / np.where(h == 0.0, 1.0, h),
                0.0,
            )
            derivative = dz - slope * radial_dot  # grad f_Sigma . u
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


__all__ = [
    "intersect_ray_with_surface",
    "solve_implicit_quadratic",
    "solve_implicit_newton",
    "solve_implicit_bracket",
    "solve_parametric",
    "intersect_rays_with_profile_surface",
]
