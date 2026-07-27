"""The boundary between two media: how a surface describes itself.

A ``Surface`` is a shape that a ray can meet. It describes itself in one of
two ways — implicitly (``f_Sigma(x) = 0``) or parametrically (``x = P(t)``)
— and that description is *all* it needs to be intersectable: the solving
is done once, generically, by
:func:`raytracer.math.intersections.intersect_ray_with_surface`. A surface
therefore owns its shape, its placement (local frame), its clear aperture,
and the media on either side; it does not own root-finding, and it does not
decide what happens after a hit (that is the propagation algorithm's job).

The implicit gradient does double duty: it is both the Newton derivative
the solver needs and the surface normal, so a surface that can state
``f_Sigma`` and ``grad f_Sigma`` gets intersection *and* normals for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike

from ..math.intersections import intersect_ray_with_surface
from ..math.vectors import as_vector, normalize

if TYPE_CHECKING:  # annotations only -- surfaces never import optics at runtime
    from ..optics.ray import Ray


class Surface:
    """Abstract base: a shape between two media, hit-testable by description.

    Subclasses provide **either** an implicit description
    (:meth:`implicit` + :meth:`implicit_gradient`, the default) **or** a
    parametric one (:meth:`parametric` + :meth:`parametric_seed`, by setting
    ``intersection_method = "parametric"``), plus optionally a solver hint
    (:attr:`quadratic_form` or :meth:`newton_seed`) to get an exact or
    faster solve than the generic bracket-and-bisect fallback.

    ``absorbing`` surfaces terminate rays: the tracer records the hit but
    spawns no children (used by instruments, baffles, and lens rims).
    """

    #: Which description this surface offers: "implicit" or "parametric".
    intersection_method: str = "implicit"

    #: ``(A, B, C, D, E, F)`` when ``f_Sigma`` is quadratic in 2-D, enabling
    #: a closed-form root instead of an iterative solve. ``None`` otherwise.
    quadratic_form: tuple[float, float, float, float, float, float] | None = None

    #: ``(lam_max, samples)`` scan window for the bracket-and-bisect fallback.
    lambda_search: tuple[float, int] = (50.0, 500)

    #: Local frame placing this surface in the world; ``None`` means the
    #: surface is already defined in world coordinates.
    frame = None

    #: Multiplies the outward normal, for surfaces with an inherited
    #: orientation convention (the tracer re-orients against the ray anyway).
    normal_sign: float = 1.0

    def __init__(
        self,
        *,
        surface_id: str = "surface",
        n_exterior: float = 1.0,
        n_interior: float = 1.0,
        absorbing: bool = False,
    ) -> None:
        self.surface_id = surface_id
        self.n_exterior = float(n_exterior)
        self.n_interior = float(n_interior)
        self.absorbing = bool(absorbing)

    # -- how this surface describes itself (in its own local frame) --------

    def implicit(self, point: np.ndarray) -> float:
        """``f_Sigma(x)``: zero exactly on the surface, signed off it."""

        raise NotImplementedError

    def implicit_gradient(self, point: np.ndarray) -> np.ndarray:
        """``grad f_Sigma(x)``: the un-normalized normal direction."""

        raise NotImplementedError

    def parametric(self, t: float) -> np.ndarray:
        """``P(t)``: the surface point at parameter *t*."""

        raise NotImplementedError

    # -- solver hints (optional) -------------------------------------------

    def newton_seed(self, origin: np.ndarray, direction: np.ndarray) -> float | None:
        """Initial ``lambda`` for the implicit Newton solve, or ``None`` to
        fall back to bracket-and-bisect."""

        return None

    def parametric_seed(self, origin: np.ndarray, direction: np.ndarray) -> tuple[float, float]:
        """Initial ``(lambda, t)`` for the parametric solve."""

        raise NotImplementedError

    # -- placement and extent ----------------------------------------------

    def within_aperture(self, point: np.ndarray) -> bool:
        """Whether a local-frame *point* lies inside the clear aperture."""

        return True

    def local_parameters(self, point: np.ndarray) -> np.ndarray:
        """Surface-specific coordinates recorded on the hit (local point by default)."""

        return np.asarray(point, dtype=float)

    def polyline(self, samples: int = 512) -> np.ndarray:
        """Vertices approximating the surface in world space."""

        raise NotImplementedError

    # -- hit testing (generic; subclasses rarely override) ------------------

    def to_local(self, point, direction: bool = False):
        if self.frame is None:
            return np.asarray(point, dtype=float)
        return (
            self.frame.direction_to_local(point) if direction else self.frame.to_local(point)
        )

    def to_world(self, point, direction: bool = False):
        if self.frame is None:
            return np.asarray(point, dtype=float)
        return (
            self.frame.direction_to_world(point) if direction else self.frame.to_world(point)
        )

    def hit(self, ray: "Ray") -> Optional["Intersection"]:
        """First intersection of *ray* with this surface, or ``None``.

        Uniform for every surface: convert into the local frame, let
        :func:`~raytracer.math.intersections.intersect_ray_with_surface`
        solve whichever description this surface offers, reject the hit if
        it falls outside the clear aperture, and build the normal from the
        implicit gradient.
        """

        origin = self.to_local(ray.origin)
        direction = self.to_local(ray.direction, direction=True)

        result = intersect_ray_with_surface(origin, direction, self)
        if result is None:
            return None
        point, lam = result
        if not self.within_aperture(point):
            return None

        normal = normalize(self.implicit_gradient(point)) * self.normal_sign
        return Intersection(
            point=self.to_world(point),
            normal=normalize(self.to_world(normal, direction=True)),
            distance=float(lam),
            surface_id=self.surface_id,
            parameters=self.local_parameters(point),
            surface=self,
        )


@dataclass
class Intersection:
    """Hit record storing the intersection between a ray and a surface."""

    point: ArrayLike
    normal: ArrayLike
    distance: float
    surface_id: str
    parameters: Optional[np.ndarray] = None
    surface: object | None = None
    meta: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        self.point = as_vector(self.point)
        self.normal = normalize(self.normal)
        if self.parameters is not None:
            self.parameters = np.asarray(self.parameters, dtype=float)
        if self.meta is None:
            self.meta = {}


__all__ = ["Surface", "Intersection"]
