"""Assemblies: elements (and other assemblies) moved together as one rigid group."""

from __future__ import annotations

from typing import Any, Iterator, Sequence, Union

from ..elements import Element
from ..geometry.transforms import Pose

Component = Union[Element, "Assembly"]


class Assembly:
    """A named rigid group of components, with a pose of its own.

    Component poses are relative to the assembly's frame, so moving the
    assembly moves everything in it — a lens barrel, a cemented group, a
    whole subsystem.
    """

    def __init__(self, name: str, components: Sequence[Component], **placement: Any) -> None:
        if not isinstance(name, str) or not name or "." in name or ":" in name:
            raise ValueError(f"an assembly has a non-empty name without '.' or ':' (got {name!r})")
        components = tuple(components)
        for c in components:
            if not isinstance(c, (Element, Assembly)):
                raise TypeError(f"an assembly holds elements and assemblies, got {c!r}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "pose", Pose.from_spec(**placement))

    def __setattr__(self, name, value):
        raise AttributeError("Assembly is immutable; use .replace(...), .placed(...), .rotated(...)")

    def walk(self, parent: Pose | None = None) -> Iterator[tuple[Element, Pose, tuple[str, ...]]]:
        """Every element inside, with its world pose and the assembly names above it."""

        here = (parent or Pose()) @ self.pose
        for c in self.components:
            if isinstance(c, Assembly):
                for element, pose, path in c.walk(here):
                    yield element, pose, (self.name, *path)
            else:
                yield c, here @ c.pose, (self.name,)

    def replace(self, **changes: Any) -> "Assembly":
        name = changes.pop("name", self.name)
        components = changes.pop("components", self.components)
        if not any(k in changes for k in ("pose", "pos", "z", "rot", "axis", "normal")):
            changes["pose"] = self.pose
        return Assembly(name, components, **changes)

    def placed(self, pose: Pose) -> "Assembly":
        return self.replace(pose=pose)

    def translated(self, offset) -> "Assembly":
        return self.placed(self.pose.translated(offset))

    def rotated(self, angle: float, axis, about=(0.0, 0.0, 0.0)) -> "Assembly":
        return self.placed(self.pose.rotated(angle, axis, about))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Assembly) and (self.name, self.components, self.pose) == (
            other.name,
            other.components,
            other.pose,
        )

    def __hash__(self) -> int:
        return hash((self.name, self.components, self.pose))

    def __repr__(self) -> str:
        return f"Assembly({self.name!r}, {len(self.components)} components, pose={self.pose!r})"


__all__ = ["Assembly", "Component"]
