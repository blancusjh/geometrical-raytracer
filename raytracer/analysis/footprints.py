"""Footprints: where and how traced rays met the hardware."""

from __future__ import annotations

import numpy as np

from ..conditions import Configuration
from ..rays import Rays
from ..system import OpticalSystem

Subject = OpticalSystem | Configuration


def _system(subject: Subject) -> OpticalSystem:
    return subject.system if isinstance(subject, Configuration) else subject


def detected_on(subject: Subject, rays: Rays, detector: str) -> tuple[np.ndarray, np.ndarray]:
    """The rays *detector* detected: where each landed, in the detector's own frame ``(K, 3)``, and the power it brought.

    *subject* is the system the rays were traced through, or a configuration of it.
    """

    here = rays.detected & (rays.boundary == detector)
    return _system(subject).face(detector).pose.to_local(rays.position[here]), rays.power[here]


def angles_of_incidence(subject: Subject, rays: Rays) -> np.ndarray:
    """``(N, V)``: the angle (degrees) between each ray and the normal of the boundary at each of its points.

    *subject* is the system the rays were traced through, or a configuration of it. NaN where a ray has no
    point, at its birth, and on a lens rim.
    """

    system = _system(subject)
    out = np.full(rays.trajectory.shape[:2], np.nan)
    for j in range(1, rays.trajectory.shape[1]):
        here, before = rays.at(j), rays.at(j - 1)
        for name in set(here.boundary) - {None}:
            face = system.face(name)
            if face.surface is None:
                continue
            m = here.boundary == name
            normal = face.pose.direction_to_world(face.surface.normal(face.pose.to_local(here.position[m])))
            cos_i = np.einsum("ij,ij->i", before.direction[m], normal)
            out[m, j] = np.degrees(np.arccos(np.clip(np.abs(cos_i), 0.0, 1.0)))
    return out


__all__ = ["detected_on", "angles_of_incidence"]
