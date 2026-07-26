"""Deprecated aliases kept for one release cycle.

Importing from this module (or using the old names re-exported at package
top level) works but emits a :class:`DeprecationWarning`.
"""

from __future__ import annotations

import warnings

from .nonseq.conics2d import ConicInterface2D
from .nonseq.ovoid2d import CartesianOvoid2D  # noqa: F401 (old import path)


def _deprecated(old: str, new: str) -> None:
    warnings.warn(
        f"{old} is deprecated; use {new} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class ConicalDioptrique(ConicInterface2D):
    """Deprecated alias of :class:`ConicInterface2D`."""

    def __post_init__(self) -> None:  # type: ignore[override]
        _deprecated("ConicalDioptrique", "ConicInterface2D")
        super().__post_init__()


__all__ = ["ConicalDioptrique", "CartesianOvoid2D"]
