"""Stigmatic ovoid lenses (SOL): trains of Cartesian surfaces sharing conjugates.

A Cartesian surface is rigorously stigmatic for exactly one pair of points.
Chaining N of them so that each surface's image point is the next surface's
object point yields a system that is rigorously stigmatic end to end — free
of spherical aberration at any aperture, with no paraxial approximation
anywhere. Silva-Lora & Torres call this a *stigmatic ovoid lens* (SOL), and
the two-surface case a *stigmatic ovoid singlet lens* (SOSL); this module is
the constructor for both.

The ontology is the industry-standard one: **media fill the regions of
space; surfaces delimit two media**. A train is therefore described as
``medium_0 → Σ_0 → medium_1 → Σ_1 → ... → medium_N`` — the same alternation
a prescription row table encodes — plus the vertex positions
``ζ_0..ζ_{N-1}`` and the conjugate chain ``d_0..d_N`` (global axial
coordinates; surface k images ``(medium_k, d_k)`` onto
``(medium_{k+1}, d_{k+1})``). A medium is anything satisfying the
:class:`~raytracer.optics.materials.Material` protocol, at whatever
fidelity the application needs:

- the **constant-index medium** (*medio de índice constante*,
  :class:`~raytracer.optics.materials.ConstantIndex`) — the simplest
  model, a single refractive index, sufficient for monochromatic work;
  bare floats passed as media are normalized to it;
- a dispersive law (:class:`~raytracer.optics.materials.SellmeierMaterial`,
  :class:`~raytracer.optics.materials.CauchyMaterial`,
  :class:`~raytracer.optics.materials.AbbeMaterial`) for real glasses and
  resins. The Cartesian shapes are built from each medium's **principal
  index at the design wavelength** — rigorous stigmatism holds exactly
  there — while the traced system carries the full dispersion, so
  chromatic analysis away from the design line is honest.

Mirrors are pure surfaces (they delimit no new medium; the tracer keeps the
current one). Modeling mirror *materials* — coatings, metals — is future
work.

For fixed end conjugates ``d_0, d_N`` the intermediate conjugates are free:
every choice is stigmatic, and that freedom is what the constraint
optimization spends (see :mod:`raytracer.optimize`). An intermediate
conjugate may be ``±inf`` — a collimated space. A surface whose *both*
neighbouring conjugates are infinite degenerates to a plane (the only shape
that maps a plane wave to a plane wave exactly), which is how a flat,
manufacturable face enters an exactly stigmatic train.

The name ``Omega`` for the two-surface lens follows the author's
``cartesian-surfaces-stl-generator`` (``OmegaLens``: two Σ curves joined
into one meridional profile); here the same object is a
:class:`StigmaticTrain` of length 2, ready for the sequential tracer rather
than for STL export.

Theory: A. Silva-Lora and R. Torres, "Aplanatism in stigmatic optical
systems," J. Opt. Soc. Am. A 37 (2020); and "Superconical aplanatic ovoid
singlet lenses," J. Opt. Soc. Am. A 37, 1155-1165 (2020), whose GOTS
parametrization :func:`raytracer.surfaces.cartesian_oval.gots_params`
transcribes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..optics.materials import AIR, ConstantIndex, Material
from ..surfaces.cartesian_oval import CartesianOvalProfile
from .rows import SurfaceRow
from .system import OpticalSystem

#: The d line: default design wavelength for index evaluation.
DESIGN_WAVELENGTH_UM = 0.58756


def _as_medium(value) -> Material:
    """Normalize a media entry: numbers become constant-index media."""

    if isinstance(value, (int, float)):
        n = float(value)
        return AIR if n == 1.0 else ConstantIndex(f"N{n:.6g}", n)
    return value


@dataclass(frozen=True)
class StigmaticTrain:
    """N Cartesian surfaces delimiting a chain of N+1 media.

    ``media`` are the space-filling media ``(medium_0, ..., medium_N)`` —
    :class:`~raytracer.optics.materials.Material` objects, or bare floats
    normalized to constant-index media; ``vertices`` are
    ``(ζ_0, ..., ζ_{N-1})``, strictly increasing from ζ_0 = 0;
    ``conjugates`` are ``(d_0, ..., d_N)`` with ``d_0`` the object and
    ``d_N`` the image, both finite — intermediates may be ``±inf`` for
    collimated spaces. ``wavelength_um`` is the design wavelength at which
    the media's principal indices define the (exact) Cartesian shapes.

    The train is stigmatic for ``(d_0, d_N)`` by construction, for every
    choice of the intermediate conjugates. It is *aplanatic* only when the
    map of :func:`raytracer.analysis.aberrations.aplanatism.aplanatism_map`
    is 1 for every ray.
    """

    media: tuple
    vertices: tuple[float, ...]
    conjugates: tuple[float, ...]
    semidiameter: float | None = None
    wavelength_um: float = DESIGN_WAVELENGTH_UM

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "media", tuple(_as_medium(m) for m in self.media)
        )
        n_surfaces = len(self.vertices)
        if len(self.media) != n_surfaces + 1:
            raise ValueError(
                f"{n_surfaces} surfaces delimit {n_surfaces + 1} media, "
                f"got {len(self.media)}"
            )
        if len(self.conjugates) != n_surfaces + 1:
            raise ValueError(
                f"{n_surfaces} vertices need {n_surfaces + 1} conjugates, "
                f"got {len(self.conjugates)}"
            )
        if any(b <= a for a, b in zip(self.vertices, self.vertices[1:])):
            raise ValueError(f"vertices must be strictly increasing: {self.vertices}")
        if self.vertices[0] != 0.0:
            raise ValueError(
                f"the first vertex must sit at ζ_0 = 0 (got {self.vertices[0]}); "
                "the tracer places the first surface at the origin, so any other "
                "convention would silently shift the conjugates"
            )
        if np.isinf(self.conjugates[0]) or np.isinf(self.conjugates[-1]):
            raise ValueError(
                "the end conjugates d_0 and d_N must be finite (use a large "
                "finite stand-in for an object or image at infinity)"
            )
        indices = self.indices
        for k, (na, nb) in enumerate(zip(indices, indices[1:])):
            if na == nb:
                raise ValueError(
                    f"surface {k} separates media of equal index n={na} at the "
                    f"design wavelength {self.wavelength_um} um; a Cartesian "
                    "surface needs an index step"
                )
        for k in range(len(self.vertices)):
            for d in (self.conjugates[k], self.conjugates[k + 1]):
                if not np.isinf(d) and d == self.vertices[k]:
                    raise ValueError(
                        f"conjugate d={d} coincides with vertex ζ_{k}; the "
                        "GOTS parameters are singular there"
                    )

    # -- derived geometry --------------------------------------------------

    @property
    def indices(self) -> tuple[float, ...]:
        """Principal refractive indices at the design wavelength, per medium."""

        return tuple(float(m.index(self.wavelength_um)) for m in self.media)

    @property
    def n_surfaces(self) -> int:
        return len(self.vertices)

    @property
    def profiles(self) -> tuple[CartesianOvalProfile, ...]:
        """The per-surface Cartesian profiles, in vertex-relative coordinates."""

        indices = self.indices
        return tuple(
            CartesianOvalProfile(
                n0=indices[k],
                z0=self.conjugates[k] - self.vertices[k],
                ni=indices[k + 1],
                zi=self.conjugates[k + 1] - self.vertices[k],
            )
            for k in range(self.n_surfaces)
        )

    def rows(self) -> list[SurfaceRow]:
        rows = []
        for k, profile in enumerate(self.profiles):
            if k + 1 < self.n_surfaces:
                thickness = self.vertices[k + 1] - self.vertices[k]
            else:
                thickness = self.conjugates[-1] - self.vertices[k]
            rows.append(
                SurfaceRow(
                    profile=profile,
                    thickness=thickness,
                    material_after=self.media[k + 1],
                    semidiameter=self.semidiameter,
                )
            )
        return rows

    def to_system(self) -> OpticalSystem:
        """The traceable system: object plane at ``d_0``, image plane at ``d_N``.

        The system carries the media and the design wavelength, so tracing
        at another wavelength sees the true dispersion (and the true
        chromatic aberration of shapes that are only exact at the design
        wavelength).
        """

        return OpticalSystem(
            self.rows(),
            object_space=self.media[0],
            object_z=self.conjugates[0],
            wavelength_um=self.wavelength_um,
        )

    @property
    def max_semidiameter(self) -> float:
        """Largest clear semidiameter every surface can support (its usable
        single-valued branch), before any vignetting consideration."""

        return min(p.max_usable_height for p in self.profiles)

    @property
    def is_dispersive(self) -> bool:
        """True when any medium's index varies with wavelength."""

        return any(not isinstance(m, ConstantIndex) for m in self.media)

    # -- the paper's magnification (Eqs. 29-30) ----------------------------

    @property
    def surface_magnifications(self) -> tuple[float, ...]:
        """``g_k = -(n_k / n_{k+1}) (d_{k+1} - ζ_k)/(d_k - ζ_k)`` per surface.

        Infinite for a surface that collimates and zero for one fed
        collimated light; the product :attr:`gt` pairs those factors off
        exactly instead of multiplying ``inf * 0``.
        """

        indices = self.indices
        return tuple(
            -(indices[k] / indices[k + 1])
            * (self.conjugates[k + 1] - self.vertices[k])
            / (self.conjugates[k] - self.vertices[k])
            for k in range(self.n_surfaces)
        )

    @property
    def gt(self) -> float:
        """Transverse magnification of the aplanatic system, ``∏ g_k``.

        An infinite intermediate conjugate ``d_j`` appears once as a
        numerator (surface j-1) and once as a denominator (surface j);
        the limit of that pair is 1, so both factors are dropped rather
        than evaluated.
        """

        indices = self.indices
        value = 1.0
        for k in range(self.n_surfaces):
            value *= -(indices[k] / indices[k + 1])
            num = self.conjugates[k + 1] - self.vertices[k]
            den = self.conjugates[k] - self.vertices[k]
            if not np.isinf(num):
                value *= num
            if not np.isinf(den):
                value /= den
        return value

    # -- constructors and variation ----------------------------------------

    @classmethod
    def singlet(
        cls,
        *,
        n: float,
        d0: float,
        d1: float,
        d2: float,
        thickness: float,
        n_outside: float = 1.0,
        semidiameter: float | None = None,
    ) -> "StigmaticTrain":
        """The Omega lens: two Cartesian surfaces sharing the conjugate ``d1``.

        Front vertex at 0, back vertex at ``thickness``; the lens medium is
        the constant-index medium ``n`` in an ``n_outside`` surround.
        Stigmatic between ``d0`` and ``d2`` for *any* ``d1`` — the
        intermediate conjugate and the thickness are the singlet's degrees
        of freedom.
        """

        return cls(
            media=(n_outside, n, n_outside),
            vertices=(0.0, thickness),
            conjugates=(d0, d1, d2),
            semidiameter=semidiameter,
        )

    @classmethod
    def symmetric_singlet(
        cls,
        *,
        n: float,
        d0: float,
        thickness: float,
        n_outside: float = 1.0,
        semidiameter: float | None = None,
    ) -> "StigmaticTrain":
        """The exactly aplanatic Omega lens: mirror symmetry about its center.

        The intermediate conjugate sits at the lens center (``d1 = ξ/2``, a
        real focus inside the glass) and the image at ``d2 = ξ - d0``, so
        the back surface is the front surface mirrored. Mirror symmetry makes
        every emergent ray the reflection of its entrance ray, and with the
        internal axis crossing the signed sines come out equal:
        ``sin u_0 / sin u_N = +1`` for **all** rays, not just paraxial ones.
        The Abbe sine condition holds identically with ``gt = +1`` (two
        internal inversions: an erect unit-magnification relay), and the
        aplanatism map of Silva-Lora & Torres is 1 ray by ray.
        """

        if d0 >= 0:
            raise ValueError("symmetric singlet expects a real object, d0 < 0")
        return cls(
            media=(n_outside, n, n_outside),
            vertices=(0.0, thickness),
            conjugates=(d0, thickness / 2.0, thickness - d0),
            semidiameter=semidiameter,
        )

    def with_conjugates(self, intermediates) -> "StigmaticTrain":
        """A copy with the intermediate conjugates ``d_1..d_{N-1}`` replaced.

        The end conjugates stay fixed — they are the specification, the
        intermediates are the degrees of freedom (this is the variation the
        constraint optimizer performs).
        """

        intermediates = tuple(float(v) for v in intermediates)
        if len(intermediates) != self.n_surfaces - 1:
            raise ValueError(
                f"{self.n_surfaces} surfaces have {self.n_surfaces - 1} "
                f"intermediate conjugates, got {len(intermediates)}"
            )
        return replace(
            self,
            conjugates=(self.conjugates[0], *intermediates, self.conjugates[-1]),
        )


__all__ = ["DESIGN_WAVELENGTH_UM", "StigmaticTrain"]
