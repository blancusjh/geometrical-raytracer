"""Where things are: rotations and poses in 3-D.

Conventions, fixed once for the whole package:

- **Frames are right-handed.** A pose maps *local* coordinates to *world*
  coordinates: ``world = R @ local + position``.
- **Angles are degrees** unless ``degrees=False`` is passed.
- **Euler angles are intrinsic x → y′ → z″.** ``Rotation.euler(a, b, c)``
  rotates by ``a`` about the body x axis, then ``b`` about the new y axis,
  then ``c`` about the newest z axis — the order of a Zemax coordinate
  break with order flag 0. As a matrix, ``R = Rx(a) @ Ry(b) @ Rz(c)``.
- **Positive angles are counter-clockwise** looking down the axis toward
  the origin (right-hand rule).
- **An element's own axis is its local +z.** ``Rotation.align_z(v)`` is the
  smallest rotation taking +z onto ``v``; when ``v`` is exactly −z, the
  half-turn is taken about the x axis.

Rotations and poses are immutable values: every operation returns a new one.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .vectors import as_vector, normalize


def _axis_matrix(axis: int, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == 1:
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True, eq=False)
class Rotation:
    """A proper rotation of 3-D space (orthonormal, determinant +1)."""

    matrix: np.ndarray

    def __post_init__(self) -> None:
        m = np.array(self.matrix, dtype=float)
        if m.shape != (3, 3):
            raise ValueError(f"a rotation matrix is 3x3, got shape {m.shape}")
        if not np.allclose(m @ m.T, np.eye(3), atol=1e-9) or np.linalg.det(m) < 0.0:
            raise ValueError("not a proper rotation: the matrix must be orthonormal with det = +1")
        m.setflags(write=False)
        object.__setattr__(self, "matrix", m)

    # -- constructors -----------------------------------------------------

    @classmethod
    def identity(cls) -> "Rotation":
        return cls(np.eye(3))

    @classmethod
    def euler(cls, x: float = 0.0, y: float = 0.0, z: float = 0.0, *, degrees: bool = True) -> "Rotation":
        """Intrinsic x → y′ → z″ rotation, ``R = Rx(x) @ Ry(y) @ Rz(z)``."""

        angles = np.radians([x, y, z]) if degrees else np.array([x, y, z], dtype=float)
        return cls(_axis_matrix(0, angles[0]) @ _axis_matrix(1, angles[1]) @ _axis_matrix(2, angles[2]))

    @classmethod
    def axis_angle(cls, axis: ArrayLike, angle: float, *, degrees: bool = True) -> "Rotation":
        """Rotation by *angle* about *axis* (right-hand rule), via Rodrigues' formula."""

        k = normalize(as_vector(axis, dim=3))
        theta = np.radians(angle) if degrees else float(angle)
        kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        return cls(np.eye(3) + np.sin(theta) * kx + (1.0 - np.cos(theta)) * (kx @ kx))

    @classmethod
    def align_z(cls, direction: ArrayLike) -> "Rotation":
        """The smallest rotation taking the local +z axis onto *direction*."""

        v = normalize(as_vector(direction, dim=3))
        z = np.array([0.0, 0.0, 1.0])
        cos_t = float(np.clip(v @ z, -1.0, 1.0))
        if cos_t > 1.0 - 1e-15:
            return cls.identity()
        if cos_t < -1.0 + 1e-15:
            return cls.axis_angle((1.0, 0.0, 0.0), 180.0)
        return cls.axis_angle(np.cross(z, v), np.degrees(np.arccos(cos_t)))

    @classmethod
    def coerce(cls, spec: Any) -> "Rotation":
        """Read the shorthand an element accepts as ``rot=``.

        - ``None`` → identity;
        - a :class:`Rotation` → itself;
        - three numbers ``(x, y, z)`` → :meth:`euler` (degrees);
        - ``(angle, (ax, ay, az))`` → :meth:`axis_angle` (degrees);
        - a 3×3 array → the matrix itself.

        The two tuple forms are told apart by their structure, never by
        their values, so the reading is unambiguous.
        """

        if spec is None:
            return cls.identity()
        if isinstance(spec, Rotation):
            return spec
        if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], Real) and not isinstance(spec[1], Real):
            return cls.axis_angle(spec[1], float(spec[0]))
        arr = np.asarray(spec, dtype=float)
        if arr.shape == (3,):
            return cls.euler(*arr)
        if arr.shape == (3, 3):
            return cls(arr)
        raise TypeError(
            f"cannot read {spec!r} as a rotation: use (x, y, z) Euler degrees, "
            "(angle, axis), a 3x3 matrix, or a Rotation"
        )

    # -- algebra ----------------------------------------------------------

    def __matmul__(self, other: "Rotation") -> "Rotation":
        """``(A @ B)`` applies ``B`` first, then ``A``."""

        return Rotation(self.matrix @ other.matrix)

    def inverse(self) -> "Rotation":
        return Rotation(self.matrix.T)

    def apply(self, vectors: ArrayLike) -> np.ndarray:
        """Rotate one vector ``(3,)`` or many ``(N, 3)``."""

        v = np.asarray(vectors, dtype=float)
        return v @ self.matrix.T

    @property
    def axis(self) -> np.ndarray:
        """Where the local +z axis points."""

        return self.matrix[:, 2].copy()

    def is_identity(self, tol: float = 1e-12) -> bool:
        return bool(np.allclose(self.matrix, np.eye(3), atol=tol))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Rotation) and bool(np.array_equal(self.matrix, other.matrix))

    def __hash__(self) -> int:
        return hash(self.matrix.tobytes())

    def __repr__(self) -> str:
        if self.is_identity():
            return "Rotation.identity()"
        return f"Rotation({np.array2string(self.matrix, precision=6)})"


@dataclass(frozen=True, eq=False)
class Pose:
    """A position and an orientation: the local frame of a placed thing."""

    position: np.ndarray = None  # type: ignore[assignment]
    rotation: Rotation = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        p = np.zeros(3) if self.position is None else np.array(as_vector(self.position, dim=3), dtype=float)
        p.setflags(write=False)
        object.__setattr__(self, "position", p)
        object.__setattr__(self, "rotation", Rotation.coerce(self.rotation))

    @classmethod
    def identity(cls) -> "Pose":
        return cls()

    @classmethod
    def from_spec(
        cls,
        *,
        pose: "Pose | None" = None,
        pos: ArrayLike | None = None,
        z: float | None = None,
        rot: Any = None,
        axis: ArrayLike | None = None,
        normal: ArrayLike | None = None,
    ) -> "Pose":
        """Read an element's placement keywords; exactly one way per quantity.

        Position: ``pos=(x, y, z)`` or ``z=`` (on the global axis).
        Orientation: ``rot=`` (see :meth:`Rotation.coerce`), or ``axis=`` /
        ``normal=`` — both name the element's local +z, the optical axis of
        a lens and the vertex normal of a single surface.
        Or a complete ``pose=``, alone.
        """

        if pose is not None:
            if any(v is not None for v in (pos, z, rot, axis, normal)):
                raise TypeError("pose= replaces pos/z/rot/axis/normal; give one or the other")
            return pose
        if pos is not None and z is not None:
            raise TypeError("give the position once: pos=(x, y, z) or z=")
        orientations = [v for v in (rot, axis, normal) if v is not None]
        if len(orientations) > 1:
            raise TypeError("give the orientation once: rot=, axis= or normal=")
        position = (0.0, 0.0, float(z)) if z is not None else pos
        if axis is not None or normal is not None:
            rotation = Rotation.align_z(axis if axis is not None else normal)
        else:
            rotation = Rotation.coerce(rot)
        return cls(position, rotation)

    # -- mapping ----------------------------------------------------------

    def to_world(self, points: ArrayLike) -> np.ndarray:
        return self.rotation.apply(points) + self.position

    def to_local(self, points: ArrayLike) -> np.ndarray:
        return (np.asarray(points, dtype=float) - self.position) @ self.rotation.matrix

    def direction_to_world(self, directions: ArrayLike) -> np.ndarray:
        return self.rotation.apply(directions)

    def direction_to_local(self, directions: ArrayLike) -> np.ndarray:
        return np.asarray(directions, dtype=float) @ self.rotation.matrix

    # -- algebra ----------------------------------------------------------

    def __matmul__(self, other: "Pose") -> "Pose":
        """``parent @ child``: the world pose of a frame given relative to *parent*."""

        return Pose(self.to_world(other.position), self.rotation @ other.rotation)

    def inverse(self) -> "Pose":
        inv = self.rotation.inverse()
        return Pose(-inv.apply(self.position), inv)

    def translated(self, offset: ArrayLike) -> "Pose":
        return Pose(self.position + as_vector(offset, dim=3), self.rotation)

    def rotated(self, angle: float, axis: ArrayLike, about: ArrayLike = (0.0, 0.0, 0.0), *, degrees: bool = True) -> "Pose":
        """This pose turned rigidly by *angle* about the world line through *about* along *axis*."""

        r = Rotation.axis_angle(axis, angle, degrees=degrees)
        pivot = as_vector(about, dim=3)
        return Pose(r.apply(self.position - pivot) + pivot, r @ self.rotation)

    @property
    def axis(self) -> np.ndarray:
        return self.rotation.axis

    def is_identity(self) -> bool:
        return bool(np.allclose(self.position, 0.0)) and self.rotation.is_identity()

    def isclose(self, other: "Pose", *, atol: float = 1e-9) -> bool:
        return bool(
            np.allclose(self.position, other.position, atol=atol)
            and np.allclose(self.rotation.matrix, other.rotation.matrix, atol=atol)
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Pose)
            and bool(np.array_equal(self.position, other.position))
            and self.rotation == other.rotation
        )

    def __hash__(self) -> int:
        return hash((self.position.tobytes(), self.rotation))

    def __repr__(self) -> str:
        parts = []
        if not np.allclose(self.position, 0.0):
            parts.append(f"position={tuple(float(v) for v in self.position)}")
        if not self.rotation.is_identity():
            parts.append(f"rotation={self.rotation!r}")
        return f"Pose({', '.join(parts)})"


__all__ = ["Rotation", "Pose"]
