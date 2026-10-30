"""Value semantics for surfaces: immutable, equal by parameters, hashable."""

from __future__ import annotations


class SurfaceValue:
    """Mixin: a surface is the tuple of its defining parameters.

    Subclasses list those parameters in ``_params``; equality and hashing
    compare the class and the parameters, and attributes cannot be reassigned
    after ``__init__`` (call :meth:`replace` for a modified copy).
    """

    _params: tuple[str, ...] = ()
    _frozen = False

    def _freeze(self) -> None:
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name, value) -> None:
        if self._frozen:
            raise AttributeError(f"{type(self).__name__} is immutable; use .replace({name}=...)")
        object.__setattr__(self, name, value)

    def _key(self) -> tuple:
        return tuple(getattr(self, p) for p in self._params)

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self._key() == other._key()  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._key()))

    def replace(self, **changes):
        """A copy with some parameters changed (and validated again)."""

        values = {p: getattr(self, p) for p in self._params}
        unknown = set(changes) - set(values)
        if unknown:
            raise TypeError(f"{type(self).__name__} has no parameter(s) {sorted(unknown)}")
        values.update(changes)
        return type(self)(**values)

    def __repr__(self) -> str:
        args = ", ".join(f"{p}={getattr(self, p)!r}" for p in self._params)
        return f"{type(self).__name__}({args})"
